# 21. モデルの重みを保存し、検証したファイルから再実行する

保存値は dtype、shape、little-endian payload の TensorRecord とする。Registry が名前を一意 ParamId と入力 slot に結び、全件の検証後にだけ候補 state を採用する。公開ファイルは長さと SHA-256 を調べてから解析する。

以下は置換・追記先を指定した実装全体である。`mode=write` はそのファイル全体を置き換え、`mode=append` は末尾へ追記する。テストは節ごとに追加し、同じ名前の関数を重複して貼らない。生成 C と共有ライブラリは runtime の一時ディレクトリに作られる。

## 1. 名前と ParamId を明示登録し、weight、bias、永続 buffer の一覧を得る。

<!-- todo:21-001 -->

まず、この章が追加するモジュールと共有テスト補助を保存する。後続の各節は、ここに掲載した `state::Registry::{validate,adopt,bind}`、`safetensor::read`、`fetch::{verified_read,fetch,load_shards}` の規則を、焦点を絞ったテストで確認する。

```rust file=src/lib.rs mode=append
pub mod fetch;
pub mod safetensor;
pub mod state;
```

```toml file=Cargo.toml mode=append
[dependencies.serde]
version = "=1.0.228"
features = ["derive"]

[dependencies.serde_json]
version = "=1.0.149"

[dependencies.sha2]
version = "=0.10.9"
```

```rust file=src/state.rs mode=write
use crate::{Buffer, Result};
use serde::{Deserialize, Serialize};
use std::{
	collections::{BTreeMap, BTreeSet},
	fs,
	path::Path,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
	F32,
	F16,
	BF16,
}
impl DType {
	pub fn bytes(self) -> usize {
		match self {
			Self::F32 => 4,
			Self::F16 | Self::BF16 => 2,
		}
	}
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorRecord {
	pub dtype: DType,
	pub shape: Vec<usize>,
	pub payload: Vec<u8>,
}
impl TensorRecord {
	pub fn elements(&self) -> Result<usize> {
		self.shape.iter().try_fold(1usize, |a, &b| {
			a.checked_mul(b).ok_or("shape overflow".into())
		})
	}
	pub fn validate(&self) -> Result<()> {
		let bytes = self
			.elements()?
			.checked_mul(self.dtype.bytes())
			.ok_or("byte overflow")?;
		if bytes != self.payload.len() {
			return Err("tensor payload length mismatch".into());
		}
		Ok(())
	}
	pub fn from_buffer(buffer: &Buffer) -> Self {
		Self {
			dtype: DType::F32,
			shape: buffer.shape.clone(),
			payload: buffer.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
		}
	}
	pub fn to_buffer(&self) -> Result<Buffer> {
		self.validate()?;
		let data = match self.dtype {
			DType::F32 => self
				.payload
				.chunks_exact(4)
				.map(|b| f32::from_le_bytes(b.try_into().unwrap()))
				.collect(),
			DType::BF16 => self
				.payload
				.chunks_exact(2)
				.map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
				.collect(),
			DType::F16 => self
				.payload
				.chunks_exact(2)
				.map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
				.collect(),
		};
		Buffer::new(self.shape.clone(), data)
	}
}
pub fn f16_to_f32(bits: u16) -> f32 {
	let sign = ((bits as u32) & 0x8000) << 16;
	let exp = (bits >> 10) & 31;
	let frac = (bits & 1023) as u32;
	let out = match exp {
		0 if frac == 0 => sign,
		0 => {
			let shift = frac.leading_zeros() - 21;
			sign | ((113 - shift) << 23) | ((frac << (shift + 13)) & 0x7fffff)
		}
		31 => sign | 0x7f800000 | (frac << 13),
		_ => sign | (((exp as u32) + 112) << 23) | (frac << 13),
	};
	f32::from_bits(out)
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spec {
	pub id: u32,
	pub slot: usize,
	pub dtype: DType,
	pub shape: Vec<usize>,
	pub trainable: bool,
}
pub type TensorSet = BTreeMap<u32, TensorRecord>;
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
	pub names: BTreeMap<String, Spec>,
}
impl Registry {
	pub fn register(&mut self, name: &str, spec: Spec) -> Result<()> {
		if self.names.contains_key(name) {
			return Err(format!("duplicate registration {name}"));
		}
		for old in self.names.values().filter(|v| v.id == spec.id) {
			if old != &spec {
				return Err(format!("inconsistent tied parameter {name}"));
			}
		}
		self.names.insert(name.into(), spec);
		Ok(())
	}
	pub fn parameters(&self) -> BTreeSet<u32> {
		self.names
			.values()
			.filter(|s| s.trainable)
			.map(|s| s.id)
			.collect()
	}
	pub fn validate(
		&self,
		loaded: &BTreeMap<String, TensorRecord>,
		partial: bool,
	) -> Result<TensorSet> {
		let mut errors = Vec::new();
		let mut result = TensorSet::new();
		for name in loaded.keys() {
			if !self.names.contains_key(name) {
				errors.push(format!("unexpected {name}"));
			}
		}
		for (name, spec) in &self.names {
			match loaded.get(name) {
				None if !partial => errors.push(format!("missing {name}")),
				None => {}
				Some(t) => {
					if let Err(e) = t.validate() {
						errors.push(format!("{name}: {e}"));
					}
					if t.shape != spec.shape {
						errors.push(format!("shape {name}"));
					}
					if t.dtype != spec.dtype {
						errors.push(format!("dtype {name}"));
					}
					if let Some(old) = result.insert(spec.id, t.clone()) {
						if old != *t {
							errors.push(format!("tied payload mismatch {name}"));
						}
					}
				}
			}
		}
		if errors.is_empty() {
			Ok(result)
		} else {
			Err(errors.join("\n"))
		}
	}
	pub fn bind(&self, state: &TensorSet) -> Result<Vec<Buffer>> {
		let max = self.names.values().map(|s| s.slot).max().unwrap_or(0);
		let mut buffers = vec![Buffer::new(vec![], vec![0.])?; max + 1];
		for spec in self.names.values() {
			let t = state
				.get(&spec.id)
				.ok_or_else(|| format!("missing ParamId {}", spec.id))?;
			if t.shape != spec.shape {
				return Err("binding shape mismatch".into());
			}
			buffers[spec.slot] = t.to_buffer()?;
		}
		Ok(buffers)
	}
	pub fn adopt(
		&self,
		live: &mut TensorSet,
		loaded: &BTreeMap<String, TensorRecord>,
		partial: bool,
	) -> Result<()> {
		let candidate = self.validate(loaded, partial)?;
		let mut merged = live.clone();
		merged.extend(candidate);
		*live = merged;
		Ok(())
	}
}
#[derive(Serialize, Deserialize)]
struct Metadata {
	version: u32,
	tensors: BTreeMap<String, Entry>,
}
#[derive(Serialize, Deserialize)]
struct Entry {
	dtype: DType,
	shape: Vec<usize>,
	offset: usize,
	length: usize,
}
pub fn save_state(directory: &Path, state: &BTreeMap<String, TensorRecord>) -> Result<()> {
	fs::create_dir_all(directory).map_err(|e| e.to_string())?;
	let mut payload = Vec::new();
	let mut tensors = BTreeMap::new();
	for (name, t) in state {
		t.validate()?;
		tensors.insert(
			name.clone(),
			Entry {
				dtype: t.dtype,
				shape: t.shape.clone(),
				offset: payload.len(),
				length: t.payload.len(),
			},
		);
		payload.extend(&t.payload);
	}
	fs::write(directory.join("tensors.bin"), payload).map_err(|e| e.to_string())?;
	let metadata = serde_json::to_vec_pretty(&Metadata {
		version: 1,
		tensors,
	})
	.map_err(|e| e.to_string())?;
	fs::write(directory.join("state.json"), metadata).map_err(|e| e.to_string())
}
pub fn load_state(directory: &Path) -> Result<BTreeMap<String, TensorRecord>> {
	let metadata = fs::read(directory.join("state.json")).map_err(|e| e.to_string())?;
	let value = crate::safetensor::unique_json(&metadata)?;
	let m: Metadata = serde_json::from_value(value).map_err(|e| e.to_string())?;
	if m.version != 1 {
		return Err("state version".into());
	}
	let payload = fs::read(directory.join("tensors.bin")).map_err(|e| e.to_string())?;
	let mut result = BTreeMap::new();
	let mut occupied = Vec::new();
	for (name, e) in m.tensors {
		let end = e.offset.checked_add(e.length).ok_or("offset overflow")?;
		let bytes = payload.get(e.offset..end).ok_or("payload range")?;
		occupied.push((e.offset, end));
		let t = TensorRecord {
			dtype: e.dtype,
			shape: e.shape,
			payload: bytes.to_vec(),
		};
		t.validate()?;
		result.insert(name, t);
	}
	occupied.sort();
	if occupied.windows(2).any(|w| w[0].1 > w[1].0) {
		return Err("overlapping payloads".into());
	}
	Ok(result)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BufferData {
	pub shape: Vec<usize>,
	pub data: Vec<f32>,
}
impl BufferData {
	pub fn buffer(&self) -> Result<Buffer> {
		Buffer::new(self.shape.clone(), self.data.clone())
	}
}
```

```rust file=src/safetensor.rs mode=write
use crate::{
	Result,
	state::{DType, TensorRecord},
};
use serde::{
	Deserialize, Deserializer,
	de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;
use std::collections::BTreeMap;
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
	fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
		struct V;
		impl<'de> Visitor<'de> for V {
			type Value = Unique;
			fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
				f.write_str("JSON without duplicate keys")
			}
			fn visit_map<A: MapAccess<'de>>(
				self,
				mut a: A,
			) -> std::result::Result<Unique, A::Error> {
				let mut map = serde_json::Map::new();
				while let Some((k, Unique(v))) = a.next_entry::<String, Unique>()? {
					if map.insert(k.clone(), v).is_some() {
						return Err(de::Error::custom(format!("duplicate JSON key {k}")));
					}
				}
				Ok(Unique(Value::Object(map)))
			}
			fn visit_seq<A: SeqAccess<'de>>(
				self,
				mut a: A,
			) -> std::result::Result<Unique, A::Error> {
				let mut values = Vec::new();
				while let Some(Unique(v)) = a.next_element()? {
					values.push(v);
				}
				Ok(Unique(Value::Array(values)))
			}
			fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
				Ok(Unique(Value::String(v.into())))
			}
			fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Unique, E> {
				Ok(Unique(Value::String(v)))
			}
			fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
				Ok(Unique(v.into()))
			}
			fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
				Ok(Unique(v.into()))
			}
			fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
				serde_json::Number::from_f64(v)
					.map(|n| Unique(Value::Number(n)))
					.ok_or_else(|| de::Error::custom("non-finite JSON"))
			}
			fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
				Ok(Unique(v.into()))
			}
			fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
				Ok(Unique(Value::Null))
			}
		}
		deserializer.deserialize_any(V)
	}
}
pub fn unique_json(bytes: &[u8]) -> Result<Value> {
	let mut de = serde_json::Deserializer::from_slice(bytes);
	let Unique(v) = Unique::deserialize(&mut de).map_err(|e| e.to_string())?;
	de.end().map_err(|e| e.to_string())?;
	Ok(v)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
	dtype: DType,
	shape: Vec<usize>,
	data_offsets: [usize; 2],
}
pub fn read(bytes: &[u8]) -> Result<BTreeMap<String, TensorRecord>> {
	let len = u64::from_le_bytes(
		bytes
			.get(..8)
			.ok_or("short safetensors prefix")?
			.try_into()
			.unwrap(),
	);
	let len = usize::try_from(len).map_err(|_| "header length overflow")?;
	if len > 100_000_000 {
		return Err("safetensors header too large".into());
	}
	let start = 8usize.checked_add(len).ok_or("header overflow")?;
	let header = unique_json(bytes.get(8..start).ok_or("short safetensors header")?)?;
	let object = header.as_object().ok_or("header must be an object")?;
	let payload = bytes.get(start..).ok_or("payload start")?;
	let mut spans = Vec::new();
	let mut result = BTreeMap::new();
	for (name, value) in object {
		if name == "__metadata__" {
			let metadata = value.as_object().ok_or("metadata object")?;
			if metadata.values().any(|v| !v.is_string()) {
				return Err("metadata values must be strings".into());
			}
			continue;
		}
		let h: Header = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
		let [a, b] = h.data_offsets;
		if b < a {
			return Err("reversed offsets".into());
		}
		let t = TensorRecord {
			dtype: h.dtype,
			shape: h.shape,
			payload: payload.get(a..b).ok_or("tensor outside payload")?.to_vec(),
		};
		t.validate()?;
		spans.push((a, b));
		result.insert(name.clone(), t);
	}
	spans.sort();
	let mut end = 0;
	for (a, b) in spans {
		if a != end {
			return Err("overlap or hole in safetensors payload".into());
		}
		end = b;
	}
	if end != payload.len() {
		return Err("unclaimed payload".into());
	}
	Ok(result)
}
```

```rust file=src/fetch.rs mode=write
use crate::{Result, state::TensorRecord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
	collections::{BTreeMap, BTreeSet},
	fs::{self, File},
	io::{Read, Write},
	path::{Component, Path},
	process::Command,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
	pub name: String,
	pub url: String,
	pub bytes: u64,
	pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
	pub revision: String,
	pub files: Vec<Asset>,
}
pub fn sha256(bytes: &[u8]) -> String {
	format!("{:x}", Sha256::digest(bytes))
}
impl Manifest {
	pub fn validate(&self) -> Result<()> {
		if self.revision.len() != 40 || !self.revision.bytes().all(|c| c.is_ascii_hexdigit()) {
			return Err("revision must be a pinned 40-digit commit".into());
		}
		let mut names = BTreeSet::new();
		for a in &self.files {
			if Path::new(&a.name).components().count() != 1
				|| !matches!(
					Path::new(&a.name).components().next(),
					Some(Component::Normal(_))
				) || !names.insert(&a.name)
			{
				return Err("unsafe or duplicate asset name".into());
			}
			if a.sha256.len() != 64 || !a.sha256.bytes().all(|c| c.is_ascii_hexdigit()) {
				return Err("invalid SHA256".into());
			}
			if !a.url.starts_with("http://127.0.0.1:")
				&& !a.url.contains(&format!("/resolve/{}/", self.revision))
			{
				return Err("URL is not pinned to manifest revision".into());
			}
		}
		Ok(())
	}
}
pub fn verified_read(path: &Path, asset: &Asset) -> Result<Vec<u8>> {
	let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
	if metadata.len() != asset.bytes {
		return Err(format!("length mismatch {}", asset.name));
	}
	let mut file = File::open(path).map_err(|e| e.to_string())?;
	let mut data = Vec::new();
	file.read_to_end(&mut data).map_err(|e| e.to_string())?;
	if sha256(&data) != asset.sha256.to_ascii_lowercase() {
		return Err(format!("digest mismatch {}", asset.name));
	}
	Ok(data)
}
pub fn fetch(cache: &Path, asset: &Asset) -> Result<Vec<u8>> {
	if Path::new(&asset.name).components().count() != 1
		|| !matches!(
			Path::new(&asset.name).components().next(),
			Some(Component::Normal(_))
		) {
		return Err("unsafe asset name".into());
	}
	fs::create_dir_all(cache).map_err(|e| e.to_string())?;
	let target = cache.join(&asset.name);
	if target.exists() {
		return verified_read(&target, asset);
	}
	let nonce = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| e.to_string())?
		.as_nanos();
	let temporary = cache.join(format!(
		".{}.{}.{}.part",
		asset.name,
		std::process::id(),
		nonce
	));
	let result = (|| {
		let mut created = File::options()
			.write(true)
			.create_new(true)
			.open(&temporary)
			.map_err(|e| e.to_string())?;
		created.write_all(&[]).map_err(|e| e.to_string())?;
		drop(created);
		let status = Command::new("curl")
			.args([
				"--fail",
				"--location",
				"--silent",
				"--show-error",
				"--max-time",
				"120",
				"--output",
			])
			.arg(&temporary)
			.arg(&asset.url)
			.status()
			.map_err(|e| e.to_string())?;
		if !status.success() {
			return Err(format!("download failed {}", asset.name));
		}
		let data = verified_read(&temporary, asset)?;
		File::open(&temporary)
			.and_then(|f| f.sync_all())
			.map_err(|e| e.to_string())?;
		fs::rename(&temporary, &target).map_err(|e| e.to_string())?;
		Ok(data)
	})();
	if result.is_err() {
		let _ = fs::remove_file(temporary);
	}
	result
}
pub fn load_shards(
	index_bytes: &[u8],
	verified: &BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, TensorRecord>> {
	let json = crate::safetensor::unique_json(index_bytes)?;
	let mapping = json
		.get("weight_map")
		.and_then(|v| v.as_object())
		.ok_or("shard weight_map missing")?;
	let files = mapping
		.values()
		.map(|v| v.as_str().ok_or("shard name must be string"))
		.collect::<std::result::Result<BTreeSet<_>, _>>()?;
	let mut loaded = BTreeMap::new();
	for file in files {
		let data = verified
			.get(file)
			.ok_or_else(|| format!("missing verified shard {file}"))?;
		for (name, tensor) in crate::safetensor::read(data)? {
			if mapping.get(&name).and_then(|v| v.as_str()) != Some(file) {
				return Err(format!("shard index mismatch {name}"));
			}
			if loaded.insert(name.clone(), tensor).is_some() {
				return Err(format!("duplicate sharded key {name}"));
			}
		}
	}
	if loaded.len() != mapping.len() {
		return Err("index keys missing from shards".into());
	}
	Ok(loaded)
}
```

```rust file=tests/model_io.rs mode=write
use chinigrad::{
	Binary, Buffer, Graph,
	fetch::{Asset, Manifest, fetch, load_shards, sha256, verified_read},
	safetensor,
	state::{DType, Registry, Spec, TensorRecord, f16_to_f32, load_state, save_state},
};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn temp(name: &str) -> PathBuf {
	let path = std::env::temp_dir().join(format!(
		"chibi-{name}-{}-{:?}",
		std::process::id(),
		std::thread::current().id()
	));
	let _ = fs::remove_dir_all(&path);
	fs::create_dir_all(&path).unwrap();
	path
}
fn tensor() -> TensorRecord {
	TensorRecord::from_buffer(&Buffer::new(vec![2], vec![2., 3.]).unwrap())
}
fn encoded(header: &str, payload: &[u8]) -> Vec<u8> {
	let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
	bytes.extend(header.as_bytes());
	bytes.extend(payload);
	bytes
}
```

`tests/model_io.rs::roundtrip_and_plan_binding` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn roundtrip_and_plan_binding() {
	let dir = temp("state");
	let mut registry = Registry::default();
	registry
		.register(
			"weight",
			Spec {
				id: 0,
				slot: 0,
				dtype: DType::F32,
				shape: vec![2],
				trainable: true,
			},
		)
		.unwrap();
	registry
		.register(
			"running",
			Spec {
				id: 1,
				slot: 1,
				dtype: DType::F32,
				shape: vec![],
				trainable: false,
			},
		)
		.unwrap();
	let tensors = BTreeMap::from([
		("weight".into(), tensor()),
		(
			"running".into(),
			TensorRecord::from_buffer(&Buffer::new(vec![], vec![7.]).unwrap()),
		),
	]);
	save_state(&dir, &tensors).unwrap();
	let loaded = load_state(&dir).unwrap();
	assert_eq!(tensors, loaded);
	let mut state = registry.validate(&loaded, false).unwrap();
	assert_eq!(registry.parameters().into_iter().collect::<Vec<_>>(), [0]);
	let mut graph = Graph::default();
	let w = graph.input(0, vec![2]).unwrap();
	let _running = graph.input(1, vec![]).unwrap();
	let row = graph.constant(0., vec![2]).unwrap();
	let ids = graph.binary(Binary::Add, w, row).unwrap();
	let a = graph.shrink(ids, vec![(0, 1)]).unwrap();
	let b = graph.shrink(ids, vec![(1, 2)]).unwrap();
	let a = graph.reshape(a, vec![]).unwrap();
	let b = graph.reshape(b, vec![]).unwrap();
	let product = graph.binary(Binary::Mul, a, b).unwrap();
	let four = graph.constant(4., vec![]).unwrap();
	let result = graph.binary(Binary::Mul, product, four).unwrap();
	let plan = graph.compile(&[result]).unwrap();
	let original = registry.validate(&tensors, false).unwrap();
	assert_eq!(
		plan.run(&registry.bind(&original).unwrap()).unwrap()[0].data,
		[24.]
	);
	assert_eq!(
		plan.run(&registry.bind(&state).unwrap()).unwrap()[0].data,
		[24.]
	);
	state.insert(
		0,
		TensorRecord::from_buffer(&Buffer::new(vec![2], vec![1., 1.25]).unwrap()),
	);
	assert_eq!(
		plan.run(&registry.bind(&state).unwrap()).unwrap()[0].data,
		[5.]
	);
	assert!(plan.source.contains("buffers[0]"));
}
```

実装箇所は `state::Registry::register` と `state::Registry::parameters` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 2. 最初は教育用の短い JSON metadata と little-endian f32 payload を保存する。

<!-- todo:21-002 -->


実装箇所は `state::save_state` と `state::load_state` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 3. dtype、shape、要素数、byte 数を検査してから新しい CPU buffer を作る。

<!-- todo:21-003 -->


実装箇所は `state::TensorRecord::{validate,to_buffer}` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 4. state 全体の検証が終わってから入れ替え、不完全な load でモデルを半分更新しない。

<!-- todo:21-004 -->


実装箇所は `state::Registry::adopt` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 5. 元の値24、load 後の24、差替え後の5を生成 C の実行結果で比較する。

<!-- todo:21-005 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 6. 8 byte の header 長、JSON header、data 部分の開始位置を検査する。

<!-- todo:21-006 -->


`tests/model_io.rs::safetensors_layout_and_corruption` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn safetensors_layout_and_corruption() {
	let payload = tensor().payload;
	let good = encoded(
		r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
		&payload,
	);
	assert_eq!(safetensor::read(&good).unwrap()["w"], tensor());
	for header in [
		r#"{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
		r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
		r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,999]}}"#,
		r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
		r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
		r#"{"w":{"dtype":"F32","shape":[18446744073709551615,8],"data_offsets":[0,8]}}"#,
	] {
		assert!(
			safetensor::read(&encoded(header, &payload)).is_err(),
			"{header}"
		);
	}
	for len in [0, 3, 7, 8, 12] {
		assert!(safetensor::read(&good[..len]).is_err());
	}
	assert!(safetensor::unique_json(br#"{"x":{"a":1,"a":2}}"#).is_err());
}
```

実装箇所は `safetensor::read` と `safetensor::unique_json` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 7. 各 tensor の `dtype`、`shape`、`data_offsets` を読み、積と byte 計算の overflow を拒否する。

<!-- todo:21-007 -->


実装箇所は `safetensor::read` と `safetensor::unique_json` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 8. payload 内の範囲、長さの一致、不正な重なり、重複した名前を検証する。

<!-- todo:21-008 -->


実装箇所は `safetensor::read` と `safetensor::unique_json` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 9. 未知 dtype と不正な header を明示エラーにし、f32 に推測変換しない。

<!-- todo:21-009 -->


実装箇所は `safetensor::read` と `safetensor::unique_json` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 10. Rust の safetensors reader と独立に作った小 fixture で byte と数値を照合する。

<!-- todo:21-010 -->


実装箇所は `safetensor::read` と `safetensor::unique_json` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 11. 必須 key の欠落、想定外 key、shape 不一致、dtype 不一致を一度に報告する。

<!-- todo:21-011 -->


`tests/model_io.rs::strict_atomic_load` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn strict_atomic_load() {
	let mut registry = Registry::default();
	registry
		.register(
			"w",
			Spec {
				id: 0,
				slot: 0,
				dtype: DType::F32,
				shape: vec![2],
				trainable: true,
			},
		)
		.unwrap();
	registry
		.register(
			"b",
			Spec {
				id: 1,
				slot: 1,
				dtype: DType::F32,
				shape: vec![1],
				trainable: true,
			},
		)
		.unwrap();
	let valid = BTreeMap::from([
		("w".into(), tensor()),
		(
			"b".into(),
			TensorRecord::from_buffer(&Buffer::new(vec![1], vec![1.]).unwrap()),
		),
	]);
	let mut live = registry.validate(&valid, false).unwrap();
	let before = live.clone();
	let corrupt = BTreeMap::from([
		(
			"w".into(),
			TensorRecord {
				dtype: DType::BF16,
				shape: vec![1],
				payload: vec![0, 0],
			},
		),
		("unexpected".into(), tensor()),
	]);
	let error = registry.adopt(&mut live, &corrupt, false).unwrap_err();
	for text in ["missing b", "unexpected unexpected", "shape w", "dtype w"] {
		assert!(error.contains(text));
	}
	assert_eq!(live, before);
	let partial = BTreeMap::from([("w".into(), tensor())]);
	registry.adopt(&mut live, &partial, true).unwrap();
}
```

実装箇所は `state::Registry::validate` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 12. strict load を標準にし、一部 load を明示要求した時だけ欠落を許す。

<!-- todo:21-012 -->


実装箇所は `state::Registry::validate` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 13. 名前を Plan の入力 slot に解決し、同じコンパイル済み Plan を再実行する。

<!-- todo:21-013 -->


実装箇所は `state::Registry::{bind,parameters}` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 14. 永続 buffer を保存と load に含め、optimizer が更新する parameter 一覧からは外す。

<!-- todo:21-014 -->


実装箇所は `state::Registry::{bind,parameters}` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 15. Metal と OpenCL では検証済み CPU state を転送し、CPU 版の出力と比較する。

<!-- todo:21-015 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 16. manifest に固定 revision を含む URL、期待 byte 数、SHA-256、保存名を記録する。

<!-- todo:21-016 -->


実装箇所は `fetch::Manifest::validate`、`fetch::verified_read`、`fetch::fetch` である。この要件に対する実行可能な確認は `model_io::strict_atomic_load` に置く。

```sh check
cargo test --offline --test model_io strict_atomic_load -- --exact
```

## 17. まずローカル HTTP fixture server から小ファイルを取得する test を作る。

<!-- todo:21-017 -->


`tests/model_io.rs::local_http_fetch` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
#[ignore]
fn local_http_fetch() {
	use std::{
		io::{Read, Write},
		net::TcpListener,
		thread,
	};
	let listener = TcpListener::bind("127.0.0.1:0").unwrap();
	let port = listener.local_addr().unwrap().port();
	let server = thread::spawn(move || {
		for request_number in 0..3 {
			let (mut stream, _) = listener.accept().unwrap();
			let mut request = [0; 2048];
			stream.read(&mut request).unwrap();
			let response = if request_number == 2 {
				b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nabc".as_slice()
			} else {
				b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".as_slice()
			};
			stream.write_all(response).unwrap();
		}
	});
	let dir = temp("http");
	let mut asset = Asset {
		name: "small.bin".into(),
		url: format!("http://127.0.0.1:{port}/asset"),
		bytes: 3,
		sha256: sha256(b"abc"),
	};
	let manifest = Manifest {
		revision: "a".repeat(40),
		files: vec![asset.clone()],
	};
	manifest.validate().unwrap();
	assert_eq!(fetch(&dir, &asset).unwrap(), b"abc");
	assert_eq!(fetch(&dir, &asset).unwrap(), b"abc");
	asset.name = "corrupt.bin".into();
	asset.sha256 = "0".repeat(64);
	assert!(fetch(&dir, &asset).is_err());
	assert!(!dir.join("corrupt.bin").exists());
	asset.name = "cut.bin".into();
	asset.url = format!("http://127.0.0.1:{port}/cut");
	asset.bytes = 9;
	assert!(fetch(&dir, &asset).unwrap_err().contains("download failed"));
	assert!(!dir.join("cut.bin").exists());
	assert!(
		fs::read_dir(&dir).unwrap().all(|e| !e
			.unwrap()
			.file_name()
			.to_string_lossy()
			.ends_with(".part"))
	);
	server.join().unwrap();
}
```

実装箇所は `fetch::Manifest::validate`、`fetch::verified_read`、`fetch::fetch` である。この要件に対する実行可能な確認は `model_io::local_http_fetch` に置く。

```sh check requires=loopback
cargo test --offline --test model_io local_http_fetch -- --exact --ignored
```

## 18. 一時ファイルへ保存し、byte 数と SHA-256 が一致した後だけ cache 名に rename する。

<!-- todo:21-018 -->


`tests/model_io.rs::digest_before_parser` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn digest_before_parser() {
	let dir = temp("digest");
	let path = dir.join("bad");
	fs::write(&path, b"not a model").unwrap();
	let asset = Asset {
		name: "bad".into(),
		url: "http://127.0.0.1:1/bad".into(),
		bytes: 11,
		sha256: "0".repeat(64),
	};
	assert!(verified_read(&path, &asset).unwrap_err().contains("digest"));
	let mut asset = asset;
	asset.sha256 = sha256(b"not a model");
	assert_eq!(verified_read(&path, &asset).unwrap(), b"not a model");
	asset.bytes += 1;
	assert!(verified_read(&path, &asset).is_err());
}
```

実装箇所は `fetch::Manifest::validate`、`fetch::verified_read`、`fetch::fetch` である。この要件に対する実行可能な確認は `model_io::digest_before_parser` に置く。

```sh check
cargo test --offline --test model_io digest_before_parser -- --exact
```

## 19. 既存 cache と直接指定されたローカルファイルも、load の前に同じ digest 検査を通す。

<!-- todo:21-019 -->


実装箇所は `fetch::Manifest::validate`、`fetch::verified_read`、`fetch::fetch` である。この要件に対する実行可能な確認は `model_io::digest_before_parser` に置く。

```sh check
cargo test --offline --test model_io digest_before_parser -- --exact
```

## 20. 通信失敗、途中切断、digest 不一致では未検証ファイルを model loader に渡さない。

<!-- todo:21-020 -->


実装箇所は `fetch::Manifest::validate`、`fetch::verified_read`、`fetch::fetch` である。この要件に対する実行可能な確認は `model_io::digest_before_parser` に置く。

```sh check
cargo test --offline --test model_io digest_before_parser -- --exact
```

## 21. F16 と BF16 の payload を読む場合は型を保持し、最初の実行用に明示的な f32 変換を用意する。

<!-- todo:21-021 -->


`tests/model_io.rs::half_precision_reference` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn half_precision_reference() {
	for (bits, value) in [
		(0, 0.),
		(0x8000, -0.),
		(0x3c00, 1.),
		(1, 2f32.powi(-24)),
		(0x7bff, 65504.),
		(0x7c00, f32::INFINITY),
		(0xfc00, f32::NEG_INFINITY),
	] {
		assert_eq!(f16_to_f32(bits).to_bits(), value.to_bits());
	}
	assert!(f16_to_f32(0x7e00).is_nan());
	for bits in [0u16, 0x8000, 0x3f80, 1, 0x7f80, 0xff80, 0x7fc1] {
		let record = TensorRecord {
			dtype: DType::BF16,
			shape: vec![1],
			payload: bits.to_le_bytes().to_vec(),
		};
		let value = record.to_buffer().unwrap().data[0];
		assert_eq!(value.to_bits(), (bits as u32) << 16);
	}
}
```

実装箇所は `state::TensorRecord::to_buffer` と `state::f16_to_f32` である。この要件に対する実行可能な確認は `model_io::half_precision_reference` に置く。

```sh check
cargo test --offline --test model_io half_precision_reference -- --exact
```

## 22. ゼロ、有限値、非正規数、無限大、NaN を使って各変換を独立の reference と比較する。

<!-- todo:21-022 -->


実装箇所は `state::TensorRecord::to_buffer` と `state::f16_to_f32` である。この要件に対する実行可能な確認は `model_io::half_precision_reference` に置く。

```sh check
cargo test --offline --test model_io half_precision_reference -- --exact
```

## 23. shard index から必要ファイルを列挙し、全 shard の digest を確認した後に state を構築する。

<!-- todo:21-023 -->


`tests/model_io.rs::shard_index_keys` を追加する。

```rust file=tests/model_io.rs mode=append
#[test]
fn shard_index_keys() {
	let payload = tensor().payload;
	let shard = encoded(
		r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
		&payload,
	);
	let verified = BTreeMap::from([("one.safetensors".into(), shard)]);
	assert_eq!(
		load_shards(br#"{"weight_map":{"w":"one.safetensors"}}"#, &verified).unwrap()["w"],
		tensor()
	);
	assert!(load_shards(br#"{"weight_map":{"wrong":"one.safetensors"}}"#, &verified).is_err());
	assert!(load_shards(br#"{"weight_map":{"w":"missing.safetensors"}}"#, &verified).is_err());
}
```

実装箇所は `fetch::load_shards` である。この要件に対する実行可能な確認は `model_io::shard_index_keys` に置く。

```sh check
cargo test --offline --test model_io shard_index_keys -- --exact
```

## 24. index 内の名前と各 shard の metadata、必要 key の集合を照合する。

<!-- todo:21-024 -->


実装箇所は `fetch::load_shards` である。この要件に対する実行可能な確認は `model_io::shard_index_keys` に置く。

```sh check
cargo test --offline --test model_io shard_index_keys -- --exact
```

## 25. config と tokenizer の各ファイルも manifest の検証対象にする。

<!-- todo:21-025 -->


実装箇所は `fetch::Manifest::validate` と22章の `model_assets::verified_files` である。この要件に対する実行可能な確認は `model_io::shard_index_keys` に置く。

```sh check
cargo test --offline --test model_io shard_index_keys -- --exact
```

## 26. f32 は roundtrip 前後の byte と数値が一致し、線形層が24を返す。

<!-- todo:21-026 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 27. SHA-256 不一致では header 解析、buffer 確保、device 転送を始めない。

<!-- todo:21-027 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::digest_before_parser` に置く。

```sh check
cargo test --offline --test model_io digest_before_parser -- --exact
```

## 28. dtype、名前、shape、offset の破損 fixture をそれぞれ拒否する。

<!-- todo:21-028 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::safetensors_layout_and_corruption` に置く。

```sh check
cargo test --offline --test model_io safetensors_layout_and_corruption -- --exact
```

## 29. dump で weight の PARAM slot を確認し、重みが定数に折り畳まれていない。

<!-- todo:21-029 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 30. device の小例は CPU reference に対し `abs_err <= 1e-5 + 1e-5*abs(ref)` とする。

<!-- todo:21-030 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```

## 31. 大きなモデルなしで、保存、load、取得、破損時の経路を検証できる。

<!-- todo:21-031 -->


実装箇所は `state::Registry::bind`、`safetensor::read`、`fetch::verified_read` である。この要件に対する実行可能な確認は `model_io::roundtrip_and_plan_binding` に置く。

```sh check
cargo test --offline --test model_io roundtrip_and_plan_binding -- --exact
```
