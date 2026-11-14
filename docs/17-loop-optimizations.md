# 17. ELFとhost queueを作り、loop・vector・CPU workerを追加する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 17.01 リンクせず ELF オブジェクトを生成

clang を ELF ターゲットで呼び出し、オブジェクトの形式、アーキテクチャ、セクション数を検査する。既存 CPU 実行は共有ライブラリのまま維持する。

`src/cpu_object.rs`を保存する。

```rust
// file="src/cpu_object.rs" mode="write"
use crate::Result;
use std::io::Write;
use std::process::{Command, Stdio};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Architecture {
    X86_64,
    Aarch64,
}

impl Architecture {
    pub(crate) fn host() -> Result<Self> {
        match std::env::consts::ARCH {
            "x86_64" => Ok(Self::X86_64),
            "aarch64" => Ok(Self::Aarch64),
            arch => Err(format!("unsupported CPU architecture {arch}")),
        }
    }

    pub(crate) fn from_machine(machine: u16) -> Result<Self> {
        match machine {
            62 => Ok(Self::X86_64),
            183 => Ok(Self::Aarch64),
            _ => Err(format!("unsupported ELF machine {machine}")),
        }
    }
}

/// Compile to an unlinked ELF object, including on macOS, where the default is Mach-O.
pub fn compile_object(source: &str) -> Result<Vec<u8>> {
    compile_for(source, Architecture::host()?)
}

pub(crate) fn compile_for(source: &str, arch: Architecture) -> Result<Vec<u8>> {
    let compiler = std::env::var_os("CC").unwrap_or_else(|| "clang".into());
    let mut command = Command::new(compiler);
    command.args([
        "-c",
        "-x",
        "c",
        "-O2",
        "-fPIC",
        "-ffreestanding",
        "-fno-math-errno",
        "-nostdlib",
        "-fno-ident",
        "-fno-stack-protector",
    ]);
    match arch {
        Architecture::X86_64 => {
            command.args(["--target=x86_64-none-unknown-elf", "-march=x86-64"]);
        }
        Architecture::Aarch64 => {
            command.args([
                "--target=aarch64-none-unknown-elf",
                "-ffixed-x18",
                "-mcpu=generic",
            ]);
        }
    }
    let mut child = command
        .args(["-", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start Clang: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("Clang stdin is unavailable")?;
    let (output, written) = std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(source.as_bytes()));
        let output = child.wait_with_output();
        let written = writer.join();
        (output, written)
    });
    let output = output.map_err(|e| format!("wait for Clang: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "Clang failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    written
        .map_err(|_| "Clang source writer panicked")?
        .map_err(|e| format!("write Clang source: {e}"))?;
    Ok(output.stdout)
}

pub fn inspect(bytes: &[u8]) -> Result<(u16, usize)> {
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" || bytes[4..7] != [2, 1, 1] {
        return Err("expected ELF64 little-endian object".into());
    }
    if u16::from_le_bytes(bytes[16..18].try_into().unwrap()) != 1 {
        return Err("expected relocatable object".into());
    }
    let machine = u16::from_le_bytes(bytes[18..20].try_into().unwrap());
    Architecture::from_machine(machine)?;
    let sections = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    Ok((machine, sections))
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod cpu_object;
```

`tests/cpu_object.rs`を保存する。

```rust
// file="tests/cpu_object.rs" mode="write"
use chibigrad::cpu_object::{compile_object, inspect};
#[test]
fn clang_produces_relocatable_object_without_linking() {
    let bytes = compile_object("void entry(void *p){*(long*)p=7;}").unwrap();
    let (machine, sections) = inspect(&bytes).unwrap();
    assert!(machine == 62 || machine == 183);
    assert!(sections > 2);
    assert!(inspect(b"bad").is_err());
    assert!(compile_object("invalid C !").is_err());
}
```

```sh
# check step=17.01
cargo test --test cpu_object
```

ELF64 の再配置可能オブジェクトが得られ、不正な C と短い入力は失敗する。

## 17.02 ELF の境界検査と配置計画

セクション、シンボル、RELA を境界付きで読み、ページ単位の読み込み配置を求める。実行可能領域を作る前に不正な表を拒否する。

`src/cpu_format.rs`を保存する。

```rust
// file="src/cpu_format.rs" mode="write"
use crate::{Result, cpu_object::Architecture};
use std::collections::HashMap;
pub(crate) const SHT_PROGBITS: u32 = 1;
pub(crate) const SHT_SYMTAB: u32 = 2;
pub(crate) const SHT_STRTAB: u32 = 3;
pub(crate) const SHT_RELA: u32 = 4;
pub(crate) const SHT_NOBITS: u32 = 8;
pub(crate) const SHT_REL: u32 = 9;
pub(crate) const SHF_WRITE: u64 = 1;
pub(crate) const SHF_ALLOC: u64 = 2;
pub(crate) const SHF_EXECINSTR: u64 = 4;
pub(crate) const SHF_TLS: u64 = 0x400;
#[derive(Debug)]
pub(crate) struct Section {
    pub(crate) name: String,
    pub(crate) kind: u32,
    pub(crate) flags: u64,
    pub(crate) offset: usize,
    pub(crate) size: usize,
    pub(crate) link: usize,
    pub(crate) info: usize,
    pub(crate) align: usize,
    pub(crate) entsize: usize,
    pub(crate) image: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct Symbol {
    pub(crate) name: String,
    pub(crate) section: usize,
    pub(crate) value: usize,
    pub(crate) weak: bool,
}

#[derive(Debug)]
pub(crate) struct Relocation {
    pub(crate) place: usize,
    pub(crate) kind: u32,
    pub(crate) symbol_table: usize,
    pub(crate) symbol: usize,
    pub(crate) addend: i64,
}

pub(crate) struct Object<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) arch: Architecture,
    pub(crate) sections: Vec<Section>,
    pub(crate) symbols: HashMap<usize, Vec<Symbol>>,
    pub(crate) relocations: Vec<Relocation>,
    pub(crate) image_size: usize,
    pub(crate) alignment: usize,
    pub(crate) trampoline_start: usize,
    pub(crate) page_size: usize,
}

pub(crate) fn bytes_at(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset.checked_add(size).ok_or("ELF offset overflow")?;
    bytes
        .get(offset..end)
        .ok_or_else(|| format!("ELF range {offset}..{end} is out of bounds"))
}
pub(crate) fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes_at(bytes, offset, 2)?.try_into().unwrap(),
    ))
}
pub(crate) fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes_at(bytes, offset, 4)?.try_into().unwrap(),
    ))
}
pub(crate) fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes_at(bytes, offset, 8)?.try_into().unwrap(),
    ))
}
pub(crate) fn usize_at(bytes: &[u8], offset: usize) -> Result<usize> {
    usize::try_from(u64_at(bytes, offset)?).map_err(|_| "ELF value does not fit usize".into())
}
pub(crate) fn string_at(bytes: &[u8], offset: usize) -> Result<String> {
    let tail = bytes
        .get(offset..)
        .ok_or("ELF string offset is out of bounds")?;
    let len = tail
        .iter()
        .position(|&v| v == 0)
        .ok_or("unterminated ELF string")?;
    String::from_utf8(tail[..len].to_vec()).map_err(|_| "ELF string is not UTF-8".into())
}
pub(crate) fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if !alignment.is_power_of_two() {
        return Err(format!("invalid ELF alignment {alignment}"));
    }
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or("ELF layout overflow".into())
}

impl<'a> Object<'a> {
    pub(crate) fn parse(bytes: &'a [u8], page_size: usize) -> Result<Self> {
        if bytes_at(bytes, 0, 4)? != b"\x7fELF" {
            return Err("missing ELF magic".into());
        }
        if bytes_at(bytes, 4, 3)? != [2, 1, 1] {
            return Err("expected ELF64 little-endian version 1".into());
        }
        bytes_at(bytes, 0, 64)?;
        if u16_at(bytes, 16)? != 1 {
            return Err("expected an unlinked ELF relocatable object".into());
        }
        let arch = Architecture::from_machine(u16_at(bytes, 18)?)?;
        if u32_at(bytes, 20)? != 1 || u16_at(bytes, 52)? != 64 {
            return Err("invalid ELF header version or size".into());
        }
        if u16_at(bytes, 54)? != 0 || u16_at(bytes, 56)? != 0 {
            return Err("unexpected ELF program headers".into());
        }
        let section_offset = usize_at(bytes, 40)?;
        let section_count = u16_at(bytes, 60)? as usize;
        let names_index = u16_at(bytes, 62)? as usize;
        if section_count == 0 || u16_at(bytes, 58)? != 64 || names_index >= section_count {
            return Err("invalid ELF section table (extended indexes are unsupported)".into());
        }
        bytes_at(
            bytes,
            section_offset,
            section_count
                .checked_mul(64)
                .ok_or("ELF section table overflow")?,
        )?;
        let mut sections = Vec::with_capacity(section_count);
        for index in 0..section_count {
            let h = section_offset + index * 64;
            let kind = u32_at(bytes, h + 4)?;
            let offset = usize_at(bytes, h + 24)?;
            let size = usize_at(bytes, h + 32)?;
            if kind != SHT_NOBITS {
                bytes_at(bytes, offset, size)?;
            }
            let alignment = usize_at(bytes, h + 48)?.max(1);
            if !alignment.is_power_of_two() {
                return Err("ELF section alignment is not a power of two".into());
            }
            sections.push(Section {
                name: String::new(),
                kind,
                flags: u64_at(bytes, h + 8)?,
                offset,
                size,
                link: u32_at(bytes, h + 40)? as usize,
                info: u32_at(bytes, h + 44)? as usize,
                align: alignment,
                entsize: usize_at(bytes, h + 56)?,
                image: None,
            });
        }
        if sections[0].kind != 0 || sections[names_index].kind != SHT_STRTAB {
            return Err("invalid ELF null section or section name table".into());
        }
        let names = bytes_at(
            bytes,
            sections[names_index].offset,
            sections[names_index].size,
        )?;
        for (index, section) in sections.iter_mut().enumerate() {
            section.name = string_at(names, u32_at(bytes, section_offset + index * 64)? as usize)?;
        }
        let mut image_size = 0usize;
        let mut alignment = page_size;
        for section in &mut sections {
            if section.flags & SHF_ALLOC == 0 {
                continue;
            }
            if section.flags & SHF_TLS != 0 {
                return Err("ELF TLS sections are unsupported".into());
            }
            if section.kind != SHT_PROGBITS && section.kind != SHT_NOBITS {
                return Err(format!(
                    "unsupported allocated ELF section {} (type {})",
                    section.name, section.kind
                ));
            }
            if section.flags & (SHF_WRITE | SHF_EXECINSTR) == (SHF_WRITE | SHF_EXECINSTR) {
                return Err("ELF section requests writable executable memory".into());
            }
            alignment = alignment.max(section.align);
            image_size = align_up(image_size, section.align.max(page_size))?;
            section.image = Some(image_size);
            image_size = image_size
                .checked_add(section.size)
                .ok_or("ELF image size overflow")?;
            image_size = align_up(image_size, page_size)?;
        }
        if image_size == 0 {
            return Err("ELF object has no allocated image".into());
        }
        let mut symbols = HashMap::new();
        for (index, section) in sections
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == SHT_SYMTAB)
        {
            if section.entsize != 24 || section.size % 24 != 0 {
                return Err("invalid ELF symbol table size".into());
            }
            let strings = sections
                .get(section.link)
                .ok_or("invalid symbol string table index")?;
            if strings.kind != SHT_STRTAB {
                return Err("symbol names do not reference a string table".into());
            }
            let strings = bytes_at(bytes, strings.offset, strings.size)?;
            let mut entries = Vec::new();
            for h in (section.offset..section.offset + section.size).step_by(24) {
                let section_index = u16_at(bytes, h + 6)? as usize;
                if section_index >= sections.len()
                    && section_index != 0xfff1
                    && section_index != 0xfff2
                {
                    return Err(format!("unsupported ELF symbol section {section_index}"));
                }
                entries.push(Symbol {
                    name: string_at(strings, u32_at(bytes, h)? as usize)?,
                    section: section_index,
                    value: usize_at(bytes, h + 8)?,
                    weak: bytes[h + 4] >> 4 == 2,
                });
            }
            symbols.insert(index, entries);
        }
        let mut relocations = Vec::new();
        for section in sections
            .iter()
            .filter(|s| s.kind == SHT_RELA || s.kind == SHT_REL)
        {
            let target = sections
                .get(section.info)
                .ok_or("invalid ELF relocation target section")?;
            let Some(image) = target.image else {
                continue;
            };
            if target.name == ".eh_frame" {
                continue;
            }
            if section.kind == SHT_REL {
                return Err("ELF REL relocations without explicit addends are unsupported".into());
            }
            if section.entsize != 24 || section.size % 24 != 0 {
                return Err("invalid ELF RELA table size".into());
            }
            let table = symbols
                .get(&section.link)
                .ok_or("invalid ELF relocation symbol table")?;
            for h in (section.offset..section.offset + section.size).step_by(24) {
                let offset = usize_at(bytes, h)?;
                let info = u64_at(bytes, h + 8)?;
                let kind = info as u32;
                let width = relocation_width(arch, kind)?;
                if offset
                    .checked_add(width)
                    .is_none_or(|end| end > target.size)
                {
                    return Err("ELF relocation exceeds its target section".into());
                }
                let symbol = (info >> 32) as usize;
                if symbol >= table.len() {
                    return Err("ELF relocation symbol index is out of bounds".into());
                }
                relocations.push(Relocation {
                    place: image + offset,
                    kind,
                    symbol_table: section.link,
                    symbol,
                    addend: u64_at(bytes, h + 16)? as i64,
                });
            }
        }
        let trampoline_start = image_size;
        image_size = image_size
            .checked_add(
                relocations
                    .len()
                    .checked_mul(32)
                    .ok_or("trampoline size overflow")?,
            )
            .ok_or("ELF image size overflow")?;
        image_size = align_up(image_size, page_size)?;
        Ok(Self {
            bytes,
            arch,
            sections,
            symbols,
            relocations,
            image_size,
            alignment,
            trampoline_start,
            page_size,
        })
    }

    pub(crate) fn entry_address(&self, entry: &str, base: usize) -> Result<usize> {
        for symbol in self.symbols.values().flatten().filter(|s| s.name == entry) {
            let Some(section) = self.sections.get(symbol.section) else {
                continue;
            };
            if symbol.section != 0
                && section.flags & SHF_EXECINSTR != 0
                && symbol.value < section.size
            {
                return Ok(base
                    + section.image.ok_or("entry section is not loaded")?
                    + symbol.value);
            }
        }
        Err(format!("ELF executable entry {entry:?} was not found"))
    }
}

fn relocation_width(arch: Architecture, kind: u32) -> Result<usize> {
    match (arch, kind) {
        (_, 0) => Ok(0),
        (Architecture::X86_64, 1) | (Architecture::Aarch64, 257) => Ok(8),
        (Architecture::X86_64, 2 | 4 | 10 | 11) => Ok(4),
        (
            Architecture::Aarch64,
            258 | 261 | 275 | 277 | 278 | 282 | 283 | 284 | 285 | 286 | 299,
        ) => Ok(4),
        _ => Err(format!("unknown {arch:?} relocation type {kind}")),
    }
}

pub fn layout(bytes: &[u8]) -> Result<(usize, usize, usize)> {
    let object = Object::parse(bytes, 4096)?;
    Ok((
        object.sections.len(),
        object.relocations.len(),
        object.image_size,
    ))
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod cpu_format;
```

`tests/cpu_format.rs`を保存する。

```rust
// file="tests/cpu_format.rs" mode="write"
use chibigrad::{cpu_format::layout, cpu_object::compile_object};
#[test]
fn object_layout_checks_tables_and_relocations() {
    let bytes = compile_object(
        "extern float sinf(float); void entry(void*p){*(float*)p=sinf(*(float*)p);}",
    )
    .unwrap();
    let (sections, relocations, size) = layout(&bytes).unwrap();
    assert!(sections > 2);
    assert!(relocations > 0);
    assert!(size >= 4096);
    for end in [0, 4, 63, bytes.len() - 1] {
        assert!(layout(&bytes[..end]).is_err());
    }
    let mut wrong = bytes.clone();
    wrong[5] = 2;
    assert!(layout(&wrong).is_err());
}
```

```sh
# check step=17.02
cargo test --test cpu_format
```

外部関数呼び出しの再配置を検出し、切断した入力と異なるエンディアンを拒否する。

## 17.03 再配置なしのオブジェクトを実行

実行領域の所有者を作り、読み書きでコピーした後にページ保護を切り替える。命令キャッシュを同期し、再配置不要なオブジェクトだけを実行する。

`src/cpu_loader.rs`を保存する。

```rust
// file="src/cpu_loader.rs" mode="write"
use crate::{
    Result,
    cpu_format::*,
    cpu_memory::*,
    cpu_object::{Architecture, compile_object},
};
use std::collections::HashMap;
/// Owns both the executable mapping and dynamic libraries referenced by its code.
pub struct ElfImage {
    mapping: Mapping,
    entry: usize,
    object: Vec<u8>,
    source: Option<String>,
    _libraries: Libraries,
}

impl ElfImage {
    pub fn compile(source: &str, entry: &str) -> Result<Self> {
        let object = compile_object(source)?;
        let mut program = Self::load(&object, entry)?;
        program.source = Some(source.into());
        Ok(program)
    }
    pub fn load(bytes: &[u8], entry: &str) -> Result<Self> {
        let page_size =
            usize::try_from(unsafe { getpagesize() }).map_err(|_| "invalid host page size")?;
        let object = Object::parse(bytes, page_size)?;
        if object.arch != Architecture::host()? {
            return Err("ELF architecture differs from the running process".into());
        }
        let libraries = Libraries::new();
        let mapping = Mapping::new(object.image_size, object.alignment)?;
        let base = mapping.base as usize;
        let entry = object.entry_address(entry, base)?;
        let mut image = vec![0u8; object.image_size];
        for section in &object.sections {
            if let Some(offset) = section.image {
                if section.kind != SHT_NOBITS {
                    image[offset..offset + section.size].copy_from_slice(bytes_at(
                        object.bytes,
                        section.offset,
                        section.size,
                    )?);
                }
            }
        }
        if !object.relocations.is_empty() {
            return Err("relocation support has not been installed".into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(image.as_ptr(), mapping.base, image.len());
        }
        mapping.protect(0, mapping.size, PROT_READ)?;
        for section in &object.sections {
            if let Some(offset) = section.image {
                let protection = PROT_READ
                    | if section.flags & SHF_EXECINSTR != 0 {
                        PROT_EXEC
                    } else if section.flags & SHF_WRITE != 0 {
                        PROT_WRITE
                    } else {
                        0
                    };
                mapping.protect(
                    offset,
                    align_up(section.size, object.page_size)?,
                    protection,
                )?;
            }
        }
        mapping.protect(
            object.trampoline_start,
            object.image_size - object.trampoline_start,
            PROT_READ | PROT_EXEC,
        )?;
        libraries.clear_cache(mapping.base, mapping.size)?;
        Ok(Self {
            mapping,
            entry,
            object: bytes.to_vec(),
            source: None,
            _libraries: libraries,
        })
    }

    /// Execute trusted native code with the caller's packed argument block.
    /// ```compile_fail
    /// fn execute(p: &chibigrad::cpu::CpuProgram) { p.run(&mut [0; 8]).unwrap(); }
    /// ```
    ///
    /// # Safety
    /// The entry must obey the packed ABI. Every encoded address must remain valid
    /// with the required alignment, capacity, aliasing and callback lifetime for the call.
    pub unsafe fn run(&self, args: &mut [u8]) -> Result<()> {
        let function: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.entry) };
        unsafe {
            function(args.as_mut_ptr());
        }
        Ok(())
    }

    pub fn object(&self) -> &[u8] {
        &self.object
    }
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
    pub fn mapped_len(&self) -> usize {
        self.mapping.size
    }
}
```

`src/cpu_memory.rs`を保存する。

```rust
// file="src/cpu_memory.rs" mode="write"
use crate::{Result, cpu_format::align_up};
use std::{
    collections::HashMap,
    ffi::{CString, c_char, c_int, c_void},
};
pub(crate) const PROT_READ: c_int = 1;
pub(crate) const PROT_WRITE: c_int = 2;
pub(crate) const PROT_EXEC: c_int = 4;

#[cfg(target_os = "macos")]
const MAP_ANONYMOUS: c_int = 0x1000;
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: c_int = 0x20;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn mprotect(addr: *mut c_void, length: usize, prot: c_int) -> c_int;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    pub(crate) fn getpagesize() -> c_int;
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

pub(crate) struct Mapping {
    pub(crate) allocation: *mut c_void,
    pub(crate) allocation_size: usize,
    pub(crate) base: *mut u8,
    pub(crate) size: usize,
}

impl Mapping {
    pub(crate) fn new(size: usize, alignment: usize) -> Result<Self> {
        let allocation_size = size
            .checked_add(alignment)
            .ok_or("executable allocation size overflow")?;
        let allocation = unsafe {
            mmap(
                std::ptr::null_mut(),
                allocation_size,
                PROT_READ | PROT_WRITE,
                2 | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if allocation as isize == -1 {
            return Err(format!("mmap: {}", std::io::Error::last_os_error()));
        }
        let base = align_up(allocation as usize, alignment)? as *mut u8;
        Ok(Self {
            allocation,
            allocation_size,
            base,
            size,
        })
    }

    pub(crate) fn protect(&self, offset: usize, size: usize, protection: c_int) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if unsafe { mprotect(self.base.add(offset).cast(), size, protection) } != 0 {
            return Err(format!("mprotect: {}", std::io::Error::last_os_error()));
        }
        Ok(())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.allocation, self.allocation_size);
        }
    }
}

pub(crate) struct Libraries {
    pub(crate) handles: Vec<*mut c_void>,
}

impl Libraries {
    pub(crate) fn new() -> Self {
        #[cfg(target_os = "macos")]
        let paths = ["/usr/lib/libSystem.B.dylib"];
        #[cfg(target_os = "linux")]
        let paths = ["libm.so.6", "libgcc_s.so.1"];
        let mut handles = Vec::new();
        for path in paths {
            let path = CString::new(path).unwrap();
            let handle = unsafe { dlopen(path.as_ptr(), 1) };
            if !handle.is_null() {
                handles.push(handle);
            }
        }
        Self { handles }
    }

    pub(crate) fn resolve(&self, name: &str, registered: &HashMap<String, usize>) -> Result<usize> {
        if let Some(&address) = registered.get(name) {
            if address == 0 {
                return Err(format!("registered symbol {name} has a null address"));
            }
            return Ok(address);
        }
        let cname = CString::new(name).map_err(|_| "symbol name contains a null byte")?;
        #[cfg(target_os = "macos")]
        let default = -2isize as *mut c_void;
        #[cfg(target_os = "linux")]
        let default = std::ptr::null_mut();
        for handle in std::iter::once(default).chain(self.handles.iter().copied()) {
            let address = unsafe { dlsym(handle, cname.as_ptr()) };
            if !address.is_null() {
                return Ok(address as usize);
            }
        }
        Err(format!("undefined ELF symbol {name}"))
    }

    pub(crate) fn clear_cache(&self, start: *mut u8, size: usize) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        {
            let address = self.resolve("__clear_cache", &HashMap::new())?;
            let clear: unsafe extern "C" fn(*mut u8, *mut u8) =
                unsafe { std::mem::transmute(address) };
            unsafe {
                clear(start, start.add(size));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = (start, size);
        Ok(())
    }
}

impl Drop for Libraries {
    fn drop(&mut self) {
        for &handle in &self.handles {
            unsafe {
                dlclose(handle);
            }
        }
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod cpu_memory;


pub mod cpu_loader;
```

`tests/cpu_mapping.rs`を保存する。

```rust
// file="tests/cpu_mapping.rs" mode="write"
use chibigrad::{cpu_loader::ElfImage, cpu_object::compile_object};
#[test]
fn owned_mapping_executes_and_rebinds_a_relocation_free_object() {
    let source = "void entry(void*raw){struct __attribute__((packed)) A{long*out;long x;}*a=raw;*a->out=a->x+7;}";
    let bytes = compile_object(source).unwrap();
    let program = ElfImage::load(&bytes, "entry").unwrap();
    drop(bytes);
    let mut result = 0i64;
    for x in [2i64, 5] {
        let mut packed = (std::ptr::addr_of_mut!(result) as u64)
            .to_ne_bytes()
            .to_vec();
        packed.extend(x.to_ne_bytes());
        unsafe { program.run(&mut packed) }.unwrap();
        assert_eq!(result, x + 7);
    }
    assert!(program.mapped_len() > 0);
    assert!(ElfImage::load(b"bad", "entry").is_err());
}
```

```sh
# check step=17.03
cargo test --test cpu_mapping
```

元のバイト列を破棄しても、同じコードで 2→9、5→12 の再束縛ができる。

## 17.04 再配置と外部シンボルで CPU 実行を移行

定数領域、BSS、外部数学関数、登録コールバック、遠い分岐のトランポリンを実装する。全機能が動いた時点で cpu.rs を ELF 実装へ置換し、暫定ローダーを削除する。

`src/cpu.rs`を保存する。

```rust
// file="src/cpu.rs" mode="write"
pub use crate::cpu_object::compile_object;
use crate::cpu_relocation::apply_relocation;
use crate::{Result, cpu_format::*, cpu_memory::*, cpu_object::Architecture};
use std::collections::HashMap;
/// Owns both the executable mapping and dynamic libraries referenced by its code.
pub struct CpuProgram {
    mapping: Mapping,
    entry: usize,
    object: Vec<u8>,
    source: Option<String>,
    _libraries: Libraries,
}

impl CpuProgram {
    pub fn compile(source: &str, entry: &str) -> Result<Self> {
        Self::compile_with_symbols(source, entry, &HashMap::new())
    }

    pub fn compile_with_symbols(
        source: &str,
        entry: &str,
        symbols: &HashMap<String, usize>,
    ) -> Result<Self> {
        let object = compile_object(source)?;
        let mut program = Self::load_with_symbols(&object, entry, symbols)?;
        program.source = Some(source.to_owned());
        Ok(program)
    }

    pub fn load(bytes: &[u8], entry: &str) -> Result<Self> {
        Self::load_with_symbols(bytes, entry, &HashMap::new())
    }

    pub fn load_with_symbols(
        bytes: &[u8],
        entry: &str,
        registered: &HashMap<String, usize>,
    ) -> Result<Self> {
        let page_size =
            usize::try_from(unsafe { getpagesize() }).map_err(|_| "invalid host page size")?;
        let object = Object::parse(bytes, page_size)?;
        if object.arch != Architecture::host()? {
            return Err("ELF architecture differs from the running process".into());
        }
        let libraries = Libraries::new();
        let mapping = Mapping::new(object.image_size, object.alignment)?;
        let base = mapping.base as usize;
        let entry = object.entry_address(entry, base)?;
        let mut image = vec![0u8; object.image_size];
        for section in &object.sections {
            if let Some(offset) = section.image {
                if section.kind != SHT_NOBITS {
                    image[offset..offset + section.size].copy_from_slice(bytes_at(
                        object.bytes,
                        section.offset,
                        section.size,
                    )?);
                }
            }
        }
        let mut trampoline = object.trampoline_start;
        for relocation in &object.relocations {
            if relocation.kind == 0 {
                continue;
            }
            let symbol = &object.symbols[&relocation.symbol_table][relocation.symbol];
            let target = object.symbol_address(symbol, base, &libraries, registered)?;
            apply_relocation(
                &mut image,
                object.arch,
                base,
                relocation,
                target,
                &mut trampoline,
            )?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(image.as_ptr(), mapping.base, image.len());
        }
        mapping.protect(0, mapping.size, PROT_READ)?;
        for section in &object.sections {
            if let Some(offset) = section.image {
                let protection = PROT_READ
                    | if section.flags & SHF_EXECINSTR != 0 {
                        PROT_EXEC
                    } else if section.flags & SHF_WRITE != 0 {
                        PROT_WRITE
                    } else {
                        0
                    };
                mapping.protect(
                    offset,
                    align_up(section.size, object.page_size)?,
                    protection,
                )?;
            }
        }
        mapping.protect(
            object.trampoline_start,
            object.image_size - object.trampoline_start,
            PROT_READ | PROT_EXEC,
        )?;
        libraries.clear_cache(mapping.base, mapping.size)?;
        Ok(Self {
            mapping,
            entry,
            object: bytes.to_vec(),
            source: None,
            _libraries: libraries,
        })
    }

    /// Execute trusted native code with the caller's packed argument block.
    /// ```compile_fail
    /// fn execute(p: &chibigrad::cpu::CpuProgram) { p.run(&mut [0; 8]).unwrap(); }
    /// ```
    ///
    /// # Safety
    /// The entry must obey the packed ABI. Every encoded address must remain valid
    /// with the required alignment, capacity, aliasing and callback lifetime for the call.
    pub unsafe fn run(&self, args: &mut [u8]) -> Result<()> {
        let function: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.entry) };
        unsafe {
            function(args.as_mut_ptr());
        }
        Ok(())
    }

    pub fn object(&self) -> &[u8] {
        &self.object
    }
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
    pub fn mapped_len(&self) -> usize {
        self.mapping.size
    }
}

impl Object<'_> {
    fn symbol_address(
        &self,
        symbol: &Symbol,
        base: usize,
        libraries: &Libraries,
        registered: &HashMap<String, usize>,
    ) -> Result<usize> {
        match symbol.section {
            0 => libraries
                .resolve(&symbol.name, registered)
                .or_else(|error| if symbol.weak { Ok(0) } else { Err(error) }),
            0xfff1 => Ok(symbol.value),
            0xfff2 => Err(format!("ELF common symbol {} is unsupported", symbol.name)),
            index => {
                let section = &self.sections[index];
                let offset = section.image.ok_or_else(|| {
                    format!("symbol {} references an unloaded section", symbol.name)
                })?;
                if symbol.value > section.size {
                    return Err(format!("symbol {} exceeds its section", symbol.name));
                }
                base.checked_add(offset)
                    .and_then(|n| n.checked_add(symbol.value))
                    .ok_or("symbol address overflow".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_object::compile_for;

    fn args(output: &mut f32, values: &[f32]) -> Vec<u8> {
        let mut bytes = (output as *mut f32 as usize).to_le_bytes().to_vec();
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    const SCALAR: &str = r#"
        static __attribute__((noinline)) float helper(float x) { return x + 2; }
        void entry(void *raw) {
          struct __attribute__((packed)) Args { float *out; } *a = raw;
          *a->out = helper(5);
        }
    "#;

    #[test]
    fn scalar_seven_and_owned_object_lifetime() {
        let source = SCALAR.to_owned();
        let object = compile_object(&source).unwrap();
        let program = CpuProgram::load(&object, "entry").unwrap();
        assert_eq!(&program.object()[..4], b"\x7fELF");
        assert!(program.source().is_none());
        drop(source);
        drop(object);
        let mut output = 0.0;
        unsafe { program.run(&mut args(&mut output, &[])) }.unwrap();
        assert_eq!(output, 7.0);
        output = -1.0;
        unsafe { program.run(&mut args(&mut output, &[])) }.unwrap();
        assert_eq!(output, 7.0);
        let compiled = CpuProgram::compile(SCALAR, "entry").unwrap();
        assert_eq!(compiled.source(), Some(SCALAR));
        assert!(compiled.mapped_len() > 0);
    }

    #[test]
    fn one_program_uses_two_runtime_inputs() {
        let program = CpuProgram::compile(
            r#"
          void entry(void *raw) {
            struct __attribute__((packed)) Args { float *out; float x, y, z; } *a = raw;
            *a->out = (a->x + a->y) * a->z;
          }
        "#,
            "entry",
        )
        .unwrap();
        let mut output = 0.0;
        unsafe { program.run(&mut args(&mut output, &[2.0, 3.0, 4.0])) }.unwrap();
        assert_eq!(output, 20.0);
        unsafe { program.run(&mut args(&mut output, &[1.0, 2.0, 4.0])) }.unwrap();
        assert_eq!(output, 12.0);
    }

    #[test]
    fn rodata_relocations_and_mutable_bss() {
        let program = CpuProgram::compile(
            r#"
          static const float values[4] = {3, 5, 7, 11};
          static unsigned invocation;
          void entry(void *raw) {
            struct __attribute__((packed)) Args { float *out; float index; } *a = raw;
            *a->out = values[(unsigned)a->index & 3] + invocation++;
          }
        "#,
            "entry",
        )
        .unwrap();
        let mut output = 0.0;
        unsafe { program.run(&mut args(&mut output, &[2.0])) }.unwrap();
        assert_eq!(output, 7.0);
        unsafe { program.run(&mut args(&mut output, &[3.0])) }.unwrap();
        assert_eq!(output, 12.0);
    }

    const MATH: &str = r#"
        extern float sinf(float);
        void entry(void *raw) {
          struct __attribute__((packed)) Args { float *out; float x; } *a = raw;
          *a->out = sinf(a->x);
        }
    "#;

    #[test]
    fn external_math_symbol_and_far_call_trampoline() {
        let program = CpuProgram::compile(MATH, "entry").unwrap();
        let mut output = 0.0;
        unsafe { program.run(&mut args(&mut output, &[1.25])) }.unwrap();
        assert!((output - 1.25f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn registered_callbacks_precede_platform_symbols() {
        extern "C" fn replacement(x: f32) -> f32 {
            x + 100.0
        }
        let symbols = HashMap::from([("sinf".to_owned(), replacement as *const () as usize)]);
        let program = CpuProgram::compile_with_symbols(MATH, "entry", &symbols).unwrap();
        let mut output = 0.0;
        unsafe { program.run(&mut args(&mut output, &[1.25])) }.unwrap();
        assert_eq!(output, 101.25);
    }

    #[test]
    fn invalid_objects_entries_and_symbols_fail() {
        for bytes in [b"not ELF".as_slice(), b"\x7fELF", &[0; 64]] {
            assert!(CpuProgram::load(bytes, "entry").is_err());
        }
        let object = compile_object(SCALAR).unwrap();
        for length in [1, 16, 63, object.len() - 1] {
            assert!(CpuProgram::load(&object[..length], "entry").is_err());
        }
        assert!(
            CpuProgram::load(&object, "missing")
                .err()
                .unwrap()
                .contains("not found")
        );
        assert!(
            CpuProgram::compile(
                "extern void missing_xyz(void); void entry(void *p) { missing_xyz(); }",
                "entry"
            )
            .err()
            .unwrap()
            .contains("undefined ELF symbol missing_xyz")
        );
        assert!(
            compile_object("this is not C")
                .unwrap_err()
                .contains("Clang failed")
        );
        let mut wrong_endian = object.clone();
        wrong_endian[5] = 2;
        assert!(
            CpuProgram::load(&wrong_endian, "entry")
                .err()
                .unwrap()
                .contains("little-endian")
        );
    }

    #[test]
    fn unknown_relocation_fails_before_execution() {
        let mut bytes = compile_object(MATH).unwrap();
        let object = Object::parse(&bytes, 4096).unwrap();
        let section = object
            .sections
            .iter()
            .find(|s| s.kind == SHT_RELA && s.name != ".rela.eh_frame")
            .unwrap();
        let offset = section.offset + 8;
        drop(object);
        bytes[offset..offset + 4].copy_from_slice(&0xffffu32.to_le_bytes());
        assert!(
            CpuProgram::load(&bytes, "entry")
                .err()
                .unwrap()
                .contains("unknown")
        );
    }

    #[test]
    fn cross_compiled_objects_parse_for_both_architectures() {
        for arch in [Architecture::X86_64, Architecture::Aarch64] {
            let bytes = compile_for(MATH, arch).unwrap();
            let object = Object::parse(&bytes, 4096).unwrap();
            assert_eq!(object.arch, arch);
            assert!(!object.relocations.is_empty());
            if arch != Architecture::host().unwrap() {
                assert!(
                    CpuProgram::load(&bytes, "entry")
                        .err()
                        .unwrap()
                        .contains("architecture differs")
                );
            }
        }
    }
}
```

`src/cpu_loader.rs`を削除する。

```rust
// file="src/cpu_loader.rs" mode="delete"

```

`src/cpu_relocation.rs`を保存する。

```rust
// file="src/cpu_relocation.rs" mode="write"
use crate::{
    Result,
    cpu_format::{Relocation, align_up, u32_at},
    cpu_object::Architecture,
};
fn write_u32(image: &mut [u8], place: usize, value: u32) {
    image[place..place + 4].copy_from_slice(&value.to_le_bytes());
}

fn signed32(value: i128) -> Result<u32> {
    i32::try_from(value)
        .map(|n| n as u32)
        .map_err(|_| format!("ELF PC-relative relocation is out of range ({value})"))
}

pub(crate) fn apply_relocation(
    image: &mut [u8],
    arch: Architecture,
    base: usize,
    relocation: &Relocation,
    target: usize,
    trampoline: &mut usize,
) -> Result<()> {
    let Relocation {
        place,
        kind,
        addend,
        ..
    } = *relocation;
    let target_addend = target as i128 + addend as i128;
    let absolute_place = base as i128 + place as i128;
    let relative = target_addend - absolute_place;
    let instr = u32_at(image, place)?;
    let word = match (arch, kind) {
        (Architecture::X86_64, 1) | (Architecture::Aarch64, 257) => {
            let value =
                u64::try_from(target_addend).map_err(|_| "absolute ELF relocation overflow")?;
            image[place..place + 8].copy_from_slice(&value.to_le_bytes());
            return Ok(());
        }
        (Architecture::X86_64, 2) | (Architecture::Aarch64, 261) => signed32(relative)?,
        (Architecture::X86_64, 4) => match signed32(relative) {
            Ok(value) => value,
            Err(_) => {
                let offset = *trampoline;
                let end = offset.checked_add(14).ok_or("trampoline offset overflow")?;
                if end > image.len() {
                    return Err("trampoline exceeds reserved image space".into());
                }
                image[offset..offset + 6].copy_from_slice(&[0xff, 0x25, 0, 0, 0, 0]);
                image[offset + 6..end].copy_from_slice(&(target as u64).to_le_bytes());
                *trampoline = end;
                signed32(offset as i128 + addend as i128 - place as i128)?
            }
        },
        (Architecture::X86_64, 10) | (Architecture::Aarch64, 258) => {
            u32::try_from(target_addend).map_err(|_| "unsigned 32-bit ELF relocation overflow")?
        }
        (Architecture::X86_64, 11) => signed32(target_addend)?,
        (Architecture::Aarch64, 275) => {
            let pages = ((target_addend & !0xfff) - (absolute_place & !0xfff)) >> 12;
            if !(-(1i128 << 20)..(1i128 << 20)).contains(&pages) {
                return Err("AArch64 ADRP relocation is out of range".into());
            }
            let immediate = pages as u32 & 0x1f_ffff;
            (instr & !((3 << 29) | (0x7ffff << 5)))
                | ((immediate & 3) << 29)
                | ((immediate >> 2) << 5)
        }
        (Architecture::Aarch64, 277 | 278 | 284 | 285 | 286 | 299) => {
            let shift = match kind {
                284 => 1,
                285 => 2,
                286 => 3,
                299 => 4,
                _ => 0,
            };
            let low = target_addend as u32 & 0xfff;
            if low & ((1 << shift) - 1) != 0 {
                return Err("unaligned AArch64 load/store relocation".into());
            }
            (instr & !(0xfff << 10)) | ((low >> shift) << 10)
        }
        (Architecture::Aarch64, 282 | 283) => {
            let branch = if relative % 4 == 0 && (-(1i128 << 27)..(1i128 << 27)).contains(&relative)
            {
                relative
            } else {
                let offset = align_up(*trampoline, 4)?;
                let end = offset.checked_add(16).ok_or("trampoline offset overflow")?;
                if end > image.len() {
                    return Err("trampoline exceeds reserved image space".into());
                }
                write_u32(image, offset, 0x58000051);
                write_u32(image, offset + 4, 0xd61f0220);
                let absolute =
                    u64::try_from(target_addend).map_err(|_| "AArch64 branch address overflow")?;
                image[offset + 8..end].copy_from_slice(&absolute.to_le_bytes());
                *trampoline = end;
                offset as i128 - place as i128
            };
            if branch % 4 != 0 || !(-(1i128 << 27)..(1i128 << 27)).contains(&branch) {
                return Err("AArch64 branch trampoline is out of range".into());
            }
            (instr & !0x03ff_ffff) | ((branch >> 2) as u32 & 0x03ff_ffff)
        }
        _ => return Err(format!("unknown {arch:?} relocation type {kind}")),
    };
    write_u32(image, place, word);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_format::{u32_at, u64_at};
    fn relocate(
        arch: Architecture,
        kind: u32,
        base: usize,
        target: usize,
        addend: i64,
        instruction: u32,
    ) -> (Vec<u8>, usize) {
        let mut image = vec![0u8; 256];
        write_u32(&mut image, 0, instruction);
        let mut trampoline = 128;
        let relocation = Relocation {
            place: 0,
            kind,
            symbol_table: 0,
            symbol: 0,
            addend,
        };
        apply_relocation(&mut image, arch, base, &relocation, target, &mut trampoline).unwrap();
        (image, trampoline)
    }

    #[test]
    fn x86_pc_relative_and_far_plt_relocations() {
        let (image, end) = relocate(Architecture::X86_64, 2, 0x1000, 0x1020, -4, 0);
        assert_eq!(u32_at(&image, 0).unwrap(), 28);
        assert_eq!(end, 128);
        let (image, end) = relocate(Architecture::X86_64, 4, 0x1000, 0x1_0000_0000, -4, 0);
        assert_eq!(u32_at(&image, 0).unwrap(), 124);
        assert_eq!(&image[128..134], &[0xff, 0x25, 0, 0, 0, 0]);
        assert_eq!(u64_at(&image, 134).unwrap(), 0x1_0000_0000);
        assert_eq!(end, 142);
    }

    #[test]
    fn aarch64_page_load_and_far_branch_relocations() {
        let (image, _) = relocate(Architecture::Aarch64, 275, 0x1000, 0x3000, 0, 0x90000000);
        assert_eq!(u32_at(&image, 0).unwrap(), 0xd0000000);
        for (kind, shift) in [(277, 0), (278, 0), (284, 1), (285, 2), (286, 3), (299, 4)] {
            let (image, _) = relocate(Architecture::Aarch64, kind, 0x1000, 0x3120, 0, 0);
            assert_eq!(u32_at(&image, 0).unwrap(), (0x120 >> shift) << 10);
        }
        for (kind, instruction) in [(282, 0x14000000), (283, 0x94000000)] {
            let (image, end) = relocate(
                Architecture::Aarch64,
                kind,
                0x1000,
                0x1_0000_0000,
                0,
                instruction,
            );
            assert_eq!(u32_at(&image, 0).unwrap(), instruction | 32);
            assert_eq!(u32_at(&image, 128).unwrap(), 0x58000051);
            assert_eq!(u32_at(&image, 132).unwrap(), 0xd61f0220);
            assert_eq!(u64_at(&image, 136).unwrap(), 0x1_0000_0000);
            assert_eq!(end, 144);
        }
    }
}
```

`src/lib.rs`を保存する。

```rust
// file="src/lib.rs" mode="write"
pub type Result<T> = std::result::Result<T, String>;
pub use uop::Axis;
pub use uop::{Arg, DType, Ops, UOp};
pub mod codegen;
pub mod cpu;
pub mod data;
pub mod effects;
pub mod functions;
pub mod pattern;
pub mod runtime;
pub mod scalar;
pub mod schedule;
pub mod tensor;
pub mod uir;
pub mod uop;
pub use runtime::{Executable, InputSpec, compile_many};
pub use tensor::{Buffer, Tensor};

pub mod math_decomp;

pub mod nn;
pub mod optim;
pub fn run_named(
    executable: &Executable,
    bindings: &std::collections::BTreeMap<String, Buffer>,
) -> Result<Vec<Buffer>> {
    let inputs = executable
        .inputs()
        .iter()
        .map(|spec| {
            let buffer = bindings
                .get(&spec.name)
                .ok_or_else(|| format!("missing binding {}", spec.name))?;
            if buffer.shape != spec.shape {
                return Err(format!("binding shape {}", spec.name));
            }
            Ok(buffer.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    executable.run(&inputs, &std::collections::HashMap::new())
}

pub mod kernels;

pub mod gpu;

#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub mod backends;

pub mod renderers;

#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub mod gpu_tensor;

#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub use gpu_tensor::{Backend,GpuExecutable};

pub mod cpu_object;

pub mod cpu_format;

pub mod cpu_memory;


pub mod cpu_relocation;
```

`src/runtime.rs`を保存する。

```rust
// file="src/runtime.rs" mode="write"
use crate::codegen::signature;
use crate::{
    Result,
    codegen::{rangeify, render},
    schedule::{Schedule, create_schedule},
    tensor::{Buffer, Tensor},
    uop::{Arg, Ops, UOp, numel},
};
use std::collections::{HashMap, HashSet};
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
}
/// Compiled signatures and lifetime plans are read-only.
/// ```compile_fail
/// fn alter(e: &mut chibigrad::Executable) { e.schedule.lifetimes.clear(); }
/// ```
pub struct Executable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    native: Vec<crate::cpu::CpuProgram>,
    input_nodes: Vec<UOp>,
    captured: HashMap<u64, Buffer>,
}
pub fn compile_many(outputs: &[Tensor]) -> Result<Executable> {
    unsafe { compile_many_with_transform(outputs, |linear| Ok(linear.clone())) }
}
/// # Safety
/// The transform must preserve buffer bounds, lifetimes, control-flow scope and
/// the packed signature. Native code generated from the result must be safe to execute.
pub unsafe fn compile_many_with_transform(
    outputs: &[Tensor],
    transform: impl Fn(&UOp) -> Result<UOp>,
) -> Result<Executable> {
    let debug = std::env::var_os("CHIBI_DEBUG").is_some();
    if debug {
        eprintln!("schedule starts");
    }
    let schedule = create_schedule(&outputs.iter().map(|x| x.uop.clone()).collect::<Vec<_>>())?;
    if debug {
        eprintln!("schedule built {} kernels", schedule.kernels.len());
    }
    let produced: HashSet<_> = schedule.lifetimes.iter().map(|l| l.buffer.id).collect();
    let mut input_nodes = vec![];
    let mut inputs = vec![];
    let mut captured = HashMap::new();
    for u in &schedule.buffers {
        if produced.contains(&u.id) {
            continue;
        }
        let Arg::Buffer { name, shape, data } = &u.arg else {
            return Err("invalid buffer argument".into());
        };
        if let Some(bits) = data {
            captured.insert(
                u.id,
                Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&b| f32::from_bits(b)).collect(),
                )?,
            );
        } else {
            input_nodes.push(u.clone());
            inputs.push(InputSpec {
                name: name.clone(),
                shape: shape.clone(),
            });
        }
    }
    let mut programs = vec![];
    let mut native = vec![];
    for (i, kernel) in schedule.kernels.iter().enumerate() {
        if debug {
            eprintln!("kernel {i} rangeify");
        }
        let linear = transform(&rangeify(&kernel.sink)?)?;
        if debug {
            eprintln!("kernel {i} render {} UOps", linear.src.len());
        }
        let entry = format!("chibi_kernel_{i}");
        let source = render(&linear, &entry)?;
        let stages = vec![
            kernel.sink.clone(),
            linear,
            UOp::new(Ops::Source, vec![], Arg::Text(source.clone())),
        ];
        let stages = {
            if debug {
                eprintln!("kernel {i} compile {} source bytes", source.len());
            }
            let compiled = crate::cpu::CpuProgram::compile(&source, &entry)?;
            let mut stages = stages;
            stages.push(UOp::new(
                Ops::Binary,
                vec![],
                Arg::Bytes(compiled.object().to_vec()),
            ));
            native.push(compiled);
            stages
        };
        programs.push(UOp::new(Ops::Program, stages, Arg::Text(entry)));
    }
    Ok(Executable {
        inputs,
        schedule,
        programs,
        native,
        input_nodes,
        captured,
    })
}
impl Executable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| p.src.iter().find(|s| s.op == Ops::Source))
            .filter_map(|s| {
                if let Arg::Text(text) = &s.arg {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn run(&self, inputs: &[Buffer], _vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let expected = node.shape()?;
            if buffer.shape != expected || buffer.data.len() != numel(&expected) {
                return Err(format!(
                    "input {} expected {:?}, received {:?} with {} elements",
                    spec.name,
                    expected,
                    buffer.shape,
                    buffer.data.len()
                ));
            }
        }
        let mut external: HashMap<u64, &Buffer> =
            self.captured.iter().map(|(&id, b)| (id, b)).collect();
        for (u, b) in self.input_nodes.iter().zip(inputs) {
            external.insert(u.id, b);
        }
        let mut sizes = vec![];
        for life in &self.schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![0.; n]).collect();
        let mut slots = slots;
        let locations: HashMap<_, _> = self
            .schedule
            .lifetimes
            .iter()
            .map(|l| (l.buffer.id, l.slot))
            .collect();
        {
            for (i, program) in self.programs.iter().enumerate() {
                let linear = &program.src[1];
                let sig = signature(linear)?;
                let count = sig.buffers.iter().map(|x| x.0).max().map_or(0, |x| x + 1);
                let mut packed = vec![0u8; count * 8];
                let mut validation_status = 0f32;
                for (slot, u) in sig.buffers {
                    let pointer = if matches!(
                        & u.arg, Arg::Buffer { name, data : Some(_), .. } if name ==
                        "__chibi_error"
                    ) {
                        &mut validation_status as *mut f32 as usize
                    } else if let Some(b) = external.get(&u.id) {
                        b.data.as_ptr() as usize
                    } else {
                        let loc = locations.get(&u.id).ok_or("missing runtime allocation")?;
                        slots[*loc].as_mut_ptr() as usize
                    };
                    packed[slot * 8..slot * 8 + 8].copy_from_slice(&(pointer as u64).to_ne_bytes());
                }
                unsafe { self.native[i].run(&mut packed) }?;
                if validation_status != 0. {
                    return Err(
                        "gather/scatter index must be finite, integral, and within bounds".into(),
                    );
                }
            }
            self.schedule
                .outputs
                .iter()
                .map(|u| {
                    let slot = locations[&u.id];
                    let shape = u.shape()?;
                    Buffer::new(shape.clone(), slots[slot][..numel(&shape)].to_vec())
                })
                .collect()
        }
    }
}
```

`src/scalar.rs` の `impl Value` 内 の `pub fn from_constant`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" within="impl Value" item="pub fn from_constant"
    pub fn from_constant(u: &UOp) -> Option<Self> {
        if u.op != Ops::Const {
            return None;
        }
        Some(match u.arg {
            Arg::Bool(v) => Self::Bool(v),
            Arg::I32(v) => Self::I32(v),
            Arg::Int(v) => Self::I64(v),
            Arg::Float(v) => Self::F32Bits((v.value() as f32).to_bits()),
            _ => return None,
        })
    }
```

`src/scalar.rs` の `impl Value` 内 の `fn unpack`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" within="impl Value" item="fn unpack"
    fn unpack(dt: DType, v: u64) -> Result<Self> {
        Ok(match dt {
            DType::Bool => Self::Bool(v != 0),
            DType::I32 => Self::I32(v as i32),
            DType::I64 => Self::I64(v as i64),
            DType::F32 => Self::F32Bits(v as u32),
            _ => return Err("not a scalar dtype".into()),
        })
    }
```

`src/scalar.rs` の `pub fn verify_node`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn verify_node"
pub fn verify_node(u: &UOp) -> Result<()> {
    if u.op == Ops::Const {
        return Value::from_constant(u)
            .filter(|_| u.src.is_empty())
            .map(|_| ())
            .ok_or("invalid scalar constant".into());
    }
    if let (Ops::Param, Arg::ScalarParam { dtype, .. }) = (u.op, &u.arg) {
        return if scalar_type(*dtype) && u.src.is_empty() {
            Ok(())
        } else {
            Err("invalid scalar parameter".into())
        };
    }
    let types: Vec<_> = u.src.iter().map(UOp::dtype).collect();
    let first = *types.first().ok_or("missing scalar operand")?;
    let unary = crate::uop::is_unary(u.op);
    let n = if u.op == Ops::Where {
        3
    } else if unary {
        1
    } else if crate::uop::is_binary(u.op) {
        2
    } else {
        return Err(format!("unsupported scalar {:?}", u.op));
    };
    if types.len() != n || !types.iter().all(|t| scalar_type(*t)) {
        return Err("scalar arity or dtype mismatch".into());
    }
    match u.op {
        Ops::Cast => {
            if !matches!(u.arg, Arg::DType(dt) if scalar_type(dt)) {
                return Err("invalid CAST target".into());
            }
        }
        Ops::Bitcast => {
            if !matches!(
                (&u.arg, first),
                (Arg::DType(DType::F32 | DType::I32), DType::I32 | DType::F32)
            ) {
                return Err("BITCAST requires equal 32-bit widths".into());
            }
        }
        Ops::Where => {
            if first != DType::Bool || types[1] != types[2] {
                return Err("WHERE dtype mismatch".into());
            }
        }
        Ops::And | Ops::Or | Ops::Xor => {
            if !matches!(first, DType::Bool | DType::I32 | DType::I64) || types[1] != first {
                return Err("bitwise dtype mismatch".into());
            }
        }
        Ops::Shl | Ops::Shr | Ops::Mod => {
            if !matches!(first, DType::I32 | DType::I64) || types[1] != first {
                return Err("integer dtype mismatch".into());
            }
        }
        Ops::Eq | Ops::Lt => {
            if types[1] != first {
                return Err("comparison dtype mismatch".into());
            }
        }
        Ops::Exp | Ops::Log | Ops::Sin | Ops::Cos | Ops::Exp2 | Ops::Log2 | Ops::Sqrt => {
            if first != DType::F32 {
                return Err("transcendental requires F32".into());
            }
        }
        _ => {
            if first == DType::Bool || types.iter().any(|t| *t != first) {
                return Err("numeric dtype mismatch".into());
            }
        }
    }
    if u.op != Ops::Cast && u.op != Ops::Bitcast && u.arg != Arg::None {
        return Err("unexpected scalar argument".into());
    }
    Ok(())
}
```

`src/scalar.rs` の `pub fn verify`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn verify"
pub fn verify(root: &UOp) -> Result<Vec<DType>> {
    let mut params = HashMap::new();
    for u in root.toposort() {
        verify_node(&u)?;
        if let Arg::ScalarParam { index, dtype } = u.arg {
            if params.insert(index, dtype).is_some_and(|old| old != dtype) {
                return Err("inconsistent parameter dtype".into());
            }
        }
        if u.src.iter().all(|s| s.op == Ops::Const) && !u.src.is_empty() {
            let v = u
                .src
                .iter()
                .map(|s| Value::from_constant(s).unwrap())
                .collect::<Vec<_>>();
            eval_alu(u.op, &u.arg, &v)?;
        }
    }
    (0..params.len())
        .map(|i| {
            params
                .get(&i)
                .copied()
                .ok_or("non-dense scalar parameter slots".into())
        })
        .collect()
}
```

`src/scalar.rs` の `fn inputs_valid`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="fn inputs_valid"
fn inputs_valid(types: &[DType], inputs: &[Value]) -> Result<()> {
    if types.len() != inputs.len() || types.iter().zip(inputs).any(|(t, v)| *t != v.dtype()) {
        return Err("scalar input signature mismatch".into());
    }
    Ok(())
}
```

`src/scalar.rs` の `pub fn eval_alu`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn eval_alu"
pub fn eval_alu(op: Ops, arg: &Arg, values: &[Value]) -> Result<Value> {
    let sources = values.iter().map(|v| v.uop()).collect();
    let node = UOp::new(op, sources, arg.clone());
    verify_node(&node)?;
    let a = values[0];
    if op == Ops::Where {
        return Ok(values[if a == Value::Bool(true) { 1 } else { 2 }]);
    }
    if matches!(op, Ops::Cast | Ops::Bitcast) && *arg == Arg::DType(a.dtype()) {
        return Ok(a);
    }
    if op == Ops::Bitcast {
        return Ok(match a {
            Value::I32(v) => Value::F32Bits(v as u32),
            Value::F32Bits(v) => Value::I32(v as i32),
            _ => unreachable!(),
        });
    }
    if op == Ops::Cast {
        let Arg::DType(dt) = arg else { unreachable!() };
        return Ok(match dt {
            DType::Bool => Value::Bool(a.number() != 0.),
            DType::F32 => Value::f32(a.number() as f32),
            DType::I32 => match a {
                Value::F32Bits(v) => {
                    let x = f32::from_bits(v);
                    if !x.is_finite() || !(-2147483648.0..2147483648.0).contains(&(x as f64)) {
                        return Err("unsafe F32 to I32 CAST".into());
                    }
                    Value::I32(x as i32)
                }
                Value::I64(v) => Value::I32(v as i32),
                _ => Value::I32(a.number() as i32),
            },
            DType::I64 => match a {
                Value::F32Bits(v) => {
                    let x = f32::from_bits(v);
                    if !x.is_finite()
                        || !(-9223372036854775808.0..9223372036854775808.0).contains(&(x as f64))
                    {
                        return Err("unsafe F32 to I64 CAST".into());
                    }
                    Value::I64(x as i64)
                }
                Value::I64(v) => Value::I64(v),
                _ => Value::I64(a.number() as i64),
            },
            _ => unreachable!(),
        });
    }
    let b = values.get(1).copied();
    if matches!(op, Ops::Eq | Ops::Lt) {
        let b = b.unwrap();
        let cmp = match (a, b) {
            (Value::I64(a), Value::I64(b)) => {
                if op == Ops::Eq {
                    a == b
                } else {
                    a < b
                }
            }
            _ => {
                if op == Ops::Eq {
                    a.number() == b.number()
                } else {
                    a.number() < b.number()
                }
            }
        };
        return Ok(Value::Bool(cmp));
    }
    Ok(match a {
        Value::Bool(a) => {
            let Value::Bool(b) = b.unwrap() else {
                unreachable!()
            };
            Value::Bool(match op {
                Ops::And => a & b,
                Ops::Or => a | b,
                Ops::Xor => a ^ b,
                _ => unreachable!(),
            })
        }
        Value::I32(a) => {
            let b = match b {
                Some(Value::I32(v)) => v,
                _ => 0,
            };
            if matches!(op, Ops::Shl | Ops::Shr) && !(0..32).contains(&b) {
                return Err("unsafe I32 shift count".into());
            }
            if matches!(op, Ops::Div | Ops::Mod) && (b == 0 || (a == i32::MIN && b == -1)) {
                return Err("unsafe I32 division".into());
            }
            Value::I32(match op {
                Ops::Add => a.wrapping_add(b),
                Ops::Sub => a.wrapping_sub(b),
                Ops::Mul => a.wrapping_mul(b),
                Ops::Neg => a.wrapping_neg(),
                Ops::And => a & b,
                Ops::Or => a | b,
                Ops::Xor => a ^ b,
                Ops::Shl => a.wrapping_shl(b as u32),
                Ops::Shr => a >> b,
                Ops::Div => crate::scalar::floor_div(a as i64, b as i64).unwrap() as i32,
                Ops::Mod => crate::scalar::floor_mod(a as i64, b as i64).unwrap() as i32,
                Ops::Max => a.max(b),
                _ => return Err("invalid I32 operation".into()),
            })
        }
        Value::I64(a) => {
            let b = match b {
                Some(Value::I64(v)) => v,
                _ => 0,
            };
            if matches!(op, Ops::Shl | Ops::Shr) && !(0..64).contains(&b) {
                return Err("unsafe I64 shift count".into());
            }
            Value::I64(
                match op {
                    Ops::Add => a.checked_add(b),
                    Ops::Sub => a.checked_sub(b),
                    Ops::Mul => a.checked_mul(b),
                    Ops::Neg => a.checked_neg(),
                    Ops::And => Some(a & b),
                    Ops::Or => Some(a | b),
                    Ops::Xor => Some(a ^ b),
                    Ops::Shl => Some(a.wrapping_shl(b as u32)),
                    Ops::Shr => Some(a >> b),
                    Ops::Div => crate::scalar::floor_div(a, b),
                    Ops::Mod => crate::scalar::floor_mod(a, b),
                    Ops::Max => Some(a.max(b)),
                    _ => None,
                }
                .ok_or("invalid or overflowing I64 operation")?,
            )
        }
        Value::F32Bits(a) => {
            let a = f32::from_bits(a);
            let b = match b {
                Some(Value::F32Bits(v)) => f32::from_bits(v),
                _ => 0.,
            };
            Value::f32(match op {
                Ops::Add => a + b,
                Ops::Sub => a - b,
                Ops::Mul => a * b,
                Ops::Div => a / b,
                Ops::Neg => -a,
                Ops::Max => {
                    if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else if a == 0. && b == 0. {
                        0.
                    } else {
                        a.max(b)
                    }
                }
                Ops::Exp => a.exp(),
                Ops::Log => a.ln(),
                Ops::Sin => a.sin(),
                Ops::Cos => a.cos(),
                Ops::Exp2 => a.exp2(),
                Ops::Log2 => a.log2(),
                Ops::Sqrt => a.sqrt(),
                _ => return Err("invalid F32 operation".into()),
            })
        }
    })
}
```

`src/scalar.rs` の `pub fn compile_with_guards`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub fn compile_with_guards"
pub fn compile_with_guards(root: &UOp, guards: &[UOp]) -> Result<CompiledScalar> {
    let root = crate::functions::inline_calls(root)?;
    let types = verify(&root)?;
    for guard in guards {
        if guard.dtype() != DType::Bool {
            return Err("guard must be Bool".into());
        }
        for u in guard.toposort() {
            verify_node(&u)?;
            if let Arg::ScalarParam { index, dtype } = u.arg {
                if types.get(index) != Some(&dtype) {
                    return Err("guard parameter signature mismatch".into());
                }
            }
        }
    }
    let all = UOp::sink(
        guards
            .iter()
            .cloned()
            .chain(std::iter::once(root.clone()))
            .collect(),
    );
    let mut mapped: HashMap<u64, UOp> = HashMap::new();
    for u in all.toposort().into_iter().filter(|u| u.op != Ops::Sink) {
        let replacement = if let Arg::ScalarParam { index, dtype } = u.arg {
            let ptr = UOp::new(
                Ops::Param,
                vec![],
                Arg::Slot {
                    index: index + 1,
                    shape: vec![1],
                },
            );
            UOp::new(Ops::Load, vec![ptr], Arg::DType(dtype))
        } else {
            u.replace(u.src.iter().map(|s| mapped[&s.id].clone()).collect())
        };
        mapped.insert(u.id, replacement);
    }
    let ptr = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![1],
        },
    );
    let store = UOp::new(
        Ops::Store,
        vec![ptr, mapped[&root.id].clone()],
        Arg::DType(root.dtype()),
    );
    let status = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: types.len() + 1,
            shape: vec![1],
        },
    );
    let mut stores: Vec<_> = guards
        .iter()
        .map(|g| {
            UOp::new(
                Ops::Store,
                vec![status.clone(), mapped[&g.id].clone()],
                Arg::DType(DType::Bool),
            )
        })
        .collect();
    stores.push(store);
    let linear = crate::codegen::linearize(&UOp::sink(stores))?;
    let source = crate::codegen::render_checked(&linear, "chibi_scalar", types.len() + 1)?;
    let program = crate::cpu::CpuProgram::compile(&source, "chibi_scalar")?;
    Ok(CompiledScalar {
        source,
        program,
        types,
        output: root.dtype(),
    })
}
```

`src/scalar.rs` の `impl CompiledScalar` 内 の `pub fn run`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" within="impl CompiledScalar" item="pub fn run"
    pub fn run(&self, inputs: &[Value]) -> Result<Value> {
        inputs_valid(&self.types, inputs)?;
        let words: Vec<_> = inputs.iter().map(|v| v.pack()).collect();
        let mut output = 0u64;
        let mut status = 0i32;
        let mut args = vec![(&mut output as *mut u64) as u64];
        args.extend(words.iter().map(|w| (w as *const u64) as u64));
        args.push((&mut status as *mut i32) as u64);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(args.as_mut_ptr().cast::<u8>(), args.len() * 8)
        };
        unsafe { self.program.run(bytes) }?;
        if status != 0 {
            return Err("unsafe native scalar operation".into());
        }
        Value::unpack(self.output, output)
    }
```

`src/scalar.rs` の `pub(crate) fn floor_mod`を置き換える。

```rust
// file="src/scalar.rs" mode="replace" item="pub(crate) fn floor_mod"
pub(crate) fn floor_mod(a: i64, b: i64) -> Option<i64> {
    let r = a.checked_rem(b)?;
    if r != 0 && (r < 0) != (b < 0) {
        r.checked_add(b)
    } else {
        Some(r)
    }
}
```

`tests/cpu_mapping.rs` の `use chibigrad::`を置き換える。

```rust
// file="tests/cpu_mapping.rs" mode="replace" item="use chibigrad::"
use chibigrad::{cpu::CpuProgram, cpu_object::compile_object};
```

`tests/cpu_mapping.rs` の `fn owned_mapping_executes_and_rebinds_a_relocation_free_object`を置き換える。

```rust
// file="tests/cpu_mapping.rs" mode="replace" item="fn owned_mapping_executes_and_rebinds_a_relocation_free_object"
fn owned_mapping_executes_and_rebinds_a_relocation_free_object() {
    let source = "void entry(void*raw){struct __attribute__((packed)) A{long*out;long x;}*a=raw;*a->out=a->x+7;}";
    let bytes = compile_object(source).unwrap();
    let program = CpuProgram::load(&bytes, "entry").unwrap();
    drop(bytes);
    let mut result = 0i64;
    for x in [2i64, 5] {
        let mut packed = (std::ptr::addr_of_mut!(result) as u64)
            .to_ne_bytes()
            .to_vec();
        packed.extend(x.to_ne_bytes());
        unsafe { program.run(&mut packed) }.unwrap();
        assert_eq!(result, x + 7);
    }
    assert!(program.mapped_len() > 0);
    assert!(CpuProgram::load(b"bad", "entry").is_err());
}
```

`tests/loop_memory.rs` の `fn masked_vector_load_and_store_execute_real_ranges`を置き換える。

```rust
// file="tests/loop_memory.rs" mode="replace" item="fn masked_vector_load_and_store_execute_real_ranges"
fn masked_vector_load_and_store_execute_real_ranges() {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![3],
        },
    );
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 1,
            shape: vec![3],
        },
    );
    let range = UOp::range(UOp::int(4), 0, Axis::Loop);
    let valid = range.lt(&UOp::int(3));
    let load = input.index(range.clone(), valid.clone()).load();
    let store = output
        .index(range.clone(), valid)
        .store(load.mul(&UOp::f32(2.)));
    let linear = UOp::linear(vec![
        range.clone(),
        store,
        UOp::new(Ops::End, vec![range], Arg::None),
    ]);
    let source = codegen::render(&linear, "vector_entry").unwrap();
    let native = CpuProgram::compile(&source, "vector_entry").unwrap();
    let input = [2.0f32, 3., 4.];
    let mut output = [0.0f32, 0., 0., 12345.];
    let mut args = (input.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { native.run(&mut args) }.unwrap();
    assert_eq!(output, [4., 6., 8., 12345.]);
}
```

`tests/memory.rs` の `fn ptr`を置き換える。

```rust
// file="tests/memory.rs" mode="replace" item="fn ptr"
fn ptr(index: usize) -> UOp {
    UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index,
            shape: vec![1],
        },
    )
}
```

`tests/memory.rs` の `fn execute`を置き換える。

```rust
// file="tests/memory.rs" mode="replace" item="fn execute"
fn execute(sink: UOp, buffers: &mut [Vec<f32>]) {
    let p = codegen::to_program(&sink, "memory").unwrap();
    let Arg::Text(source) = &p.src[2].arg else {
        panic!()
    };
    let native = CpuProgram::compile(source, "memory").unwrap();
    let mut args = buffers
        .iter_mut()
        .flat_map(|b| (b.as_mut_ptr() as u64).to_ne_bytes())
        .collect::<Vec<_>>();
    unsafe { native.run(&mut args) }.unwrap();
}
```

`tests/memory.rs` の `fn scalar_memory_and_mask`を置き換える。

```rust
// file="tests/memory.rs" mode="replace" item="fn scalar_memory_and_mask"
fn scalar_memory_and_mask() {
    let x = ptr(0);
    let y = ptr(1);
    let value = x
        .index(UOp::int(0), UOp::bool(true))
        .load()
        .mul(&UOp::f32(2.));
    let masked = x.index(UOp::int(999), UOp::bool(false)).load();
    let stores = vec![
        y.index(UOp::int(0), UOp::bool(true))
            .store(value.add(&masked)),
        y.index(UOp::int(999), UOp::bool(false))
            .store(UOp::f32(99.)),
    ];
    let mut buffers = vec![vec![3.], vec![0., 12345.]];
    execute(UOp::sink(stores), &mut buffers);
    assert_eq!(buffers[1], [6., 12345.]);
}
```

`tests/reduction.rs` の `fn row_sum`を置き換える。

```rust
// file="tests/reduction.rs" mode="replace" item="fn row_sum"
fn row_sum(width: usize, values: &[f32]) -> Vec<f32> {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![2, width],
        },
    );
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 1,
            shape: vec![2],
        },
    );
    let row = UOp::range(UOp::int(2), 0, Axis::Loop);
    let accumulator = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = accumulator.store(UOp::f32(0.));
    let col = UOp::range(UOp::int(width as i64), 1, Axis::Reduce);
    let index = row.mul(&UOp::int(width as i64)).add(&col);
    let value = input.index(index, UOp::bool(true)).load();
    let state = accumulator.after(vec![init.clone(), col.clone()]);
    let update = state.store(state.load().add(&value));
    let end = UOp::new(Ops::End, vec![update.clone(), col.clone()], Arg::None);
    let result = accumulator.after(vec![end.clone()]).load();
    let store = output
        .index(row.clone(), UOp::bool(true))
        .store(result.clone());
    let outer_end = UOp::new(Ops::End, vec![store.clone(), row.clone()], Arg::None);
    let linear = UOp::linear(vec![
        row,
        accumulator,
        init,
        col,
        update,
        end,
        result,
        store,
        outer_end,
    ]);
    let source = codegen::render(&linear, "row_sum").unwrap();
    let native = CpuProgram::compile(&source, "row_sum").unwrap();
    let mut output = vec![0.0f32; 2];
    let mut args = (values.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { native.run(&mut args) }.unwrap();
    output
}
```

`tests/reduction.rs` の `fn load_store_after_form_reduction_and_reset_accumulator`を置き換える。

```rust
// file="tests/reduction.rs" mode="replace" item="fn load_store_after_form_reduction_and_reset_accumulator"
fn load_store_after_form_reduction_and_reset_accumulator() {
    assert_eq!(row_sum(3, &[1., 2., 3., 4., 5., 6.]), [6., 15.]);
    assert_eq!(row_sum(0, &[]), [0., 0.]);
}
```

`tests/reduction_max.rs` の `fn max_reduction_and_empty_identity`を置き換える。

```rust
// file="tests/reduction_max.rs" mode="replace" item="fn max_reduction_and_empty_identity"
fn max_reduction_and_empty_identity() {
    for (width, data, expected) in [
        (3, vec![-7.0f32, -2., -9., 4., 3., 2.], vec![-2., 4.]),
        (0, vec![], vec![f32::NEG_INFINITY; 2]),
    ] {
        let p = |index, n| {
            UOp::new(
                Ops::Param,
                vec![],
                Arg::Slot {
                    index,
                    shape: vec![n],
                },
            )
        };
        let linear = row_reduce(p(0, 2 * width), p(1, 2), 2, width, Ops::Max).unwrap();
        let source = codegen::render(&linear, "maximum").unwrap();
        let program = CpuProgram::compile(&source, "maximum").unwrap();
        let mut out = vec![0f32; 2];
        let mut args = (data.as_ptr() as u64).to_ne_bytes().to_vec();
        args.extend((out.as_mut_ptr() as u64).to_ne_bytes());
        unsafe { program.run(&mut args) }.unwrap();
        assert_eq!(out, expected);
    }
}
```

```sh
# check step=17.04
cargo test --lib cpu
cargo test --test cpu_mapping
```

CPU の 10 テストと所有領域の再束縛テストが通る。unsafe な実行契約と非公開メタデータを維持する。

## 17.05 ホストキューも同じ UOp でコンパイル

外部 CALL を共通 UOp と UIR に追加する。Rust ホストが直接回していた転送と送信を HWQueue の LINEAR へ移し、CPU ELF コードからネイティブコールバックを呼ぶ。GPU の計算内容は維持する。

`src/codegen.rs` の `fn render_impl`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn render_impl"
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects, scopes)
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End {
            if let Some(parent) = scopes.pop() {
                *seen = parent;
            }
        }
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects, &mut scopes)
    }
    let complete = UOp::linear(all);
    let linear = &complete;
    verify_lowered(linear)?;

    if !identifier(entry) {
        return Err("invalid C entry identifier".into());
    }
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    let mut functions: HashMap<String, (DType, Vec<DType>)> = HashMap::new();
    for u in &linear.src {
        let (body, args) = if u.op == Ops::Call {
            let b = u.src.first().ok_or("CALL lacks body")?;
            (b, &u.src[1..])
        } else if u.op == Ops::CustomFunction && !u.src.is_empty() {
            (u, &u.src[..])
        } else {
            continue;
        };
        let Arg::Function { name, dtype } = &body.arg else {
            continue;
        };
        if !identifier(name) {
            return Err("invalid C function identifier".into());
        }
        let types = args.iter().map(UOp::dtype).collect::<Vec<_>>();
        if functions
            .get(name)
            .is_some_and(|old| old != &(*dtype, types.clone()))
        {
            return Err(format!("inconsistent callback signature for {name}"));
        }
        functions.insert(name.clone(), (*dtype, types));
    }
    let mut function_names = functions.keys().collect::<Vec<_>>();
    function_names.sort();
    for name in function_names {
        let (ret, args) = &functions[name];
        text.push_str(&format!(
            "extern {} {}({});\n",
            ctype(*ret),
            name,
            if args.is_empty() {
                "void".into()
            } else {
                args.iter().map(|d| ctype(*d)).collect::<Vec<_>>().join(",")
            }
        ));
    }
    text.push_str(&format!("void {entry}(unsigned char *args) {{\n"));
    let mut values = HashMap::new();
    let mut addresses: HashMap<u64, (String, String, String, DType)> = HashMap::new();
    for (slot, u) in &sig.buffers {
        let name = format!("b{}", u.id);
        text.push_str(&format!(
            "  void *{name}=(void*)(unsigned long long)((unsigned long long*)args)[{slot}];\n"
        ));
        values.insert(u.id, name);
    }
    let mut depth = 1;
    let mut occurrence = 0usize;
    let mut value_scopes = vec![];
    let mut address_scopes = vec![];
    for u in &linear.src {
        let get = |s: &UOp| -> Result<String> {
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("{:?} {} used before definition", s.op, s.id))
        };
        let at = |i: usize| -> Result<String> { get(u.src.get(i).ok_or("missing source")?) };
        let indent = "  ".repeat(depth);
        if let Some(slot) = status_slot {
            let reject = format!(
                "{{ ((int*)(unsigned long long)((unsigned long long*)args)[{slot}])[0]=1; return; }}"
            );
            let a = || at(0);
            let b = || at(1);
            let guard = match u.op {
                Ops::Shl | Ops::Shr => Some(format!(
                    "({}<0 || {}>={})",
                    b()?,
                    b()?,
                    if u.dtype() == DType::I32 { 32 } else { 64 }
                )),
                Ops::Div | Ops::Mod if matches!(u.dtype(), DType::I32 | DType::I64) => {
                    Some(format!(
                        "({}==0 || ({}=={} && {}==-1))",
                        b()?,
                        a()?,
                        if u.dtype() == DType::I32 {
                            "(-2147483647-1)"
                        } else {
                            "(-9223372036854775807ll-1)"
                        },
                        b()?
                    ))
                }
                Ops::Cast
                    if u.src[0].dtype() == DType::F32
                        && matches!(u.dtype(), DType::I32 | DType::I64) =>
                {
                    let limit = if u.dtype() == DType::I32 {
                        "2147483648.0f"
                    } else {
                        "9223372036854775808.0f"
                    };
                    Some(format!("!({} >= -{limit} && {} < {limit})", a()?, a()?))
                }
                Ops::Add | Ops::Sub | Ops::Mul if u.dtype() == DType::I64 => {
                    let operation = match u.op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    text.push_str(&format!("{indent}long long checked{};\n", u.id));
                    Some(format!(
                        "__builtin_{operation}_overflow({}, {}, &checked{})",
                        a()?,
                        b()?,
                        u.id
                    ))
                }
                Ops::Neg if u.dtype() == DType::I64 => {
                    Some(format!("({}==(-9223372036854775807ll-1))", a()?))
                }
                _ => None,
            };
            if let Some(condition) = guard {
                text.push_str(&format!("{indent}if ({condition}) {reject}\n"));
            }
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(bits)) => {
                let bits = (bits.value() as f32).to_bits();
                values.insert(u.id, format!("chibi_float({bits}u)"));
                continue;
            }
            (Ops::Const, Arg::I32(v)) => {
                values.insert(u.id, format!("((int){}u)", *v as u32));
                continue;
            }
            (Ops::Const, Arg::Int(v)) => {
                values.insert(u.id, format!("((long long){}ull)", *v as u64));
                continue;
            }
            (Ops::Const, Arg::Bool(v)) => {
                values.insert(u.id, if *v { "1" } else { "0" }.into());
                continue;
            }
            (Ops::Buffer, _) | (Ops::Param, _) => continue,
            (Ops::Alloc, Arg::Allocation { dtype, size, .. }) => {
                let name = format!("a{}", u.id);
                text.push_str(&format!(
                    "{indent}{} {name}[{}];\n",
                    ctype(*dtype),
                    size.max(&1)
                ));
                values.insert(u.id, name);
                continue;
            }
            (Ops::After, _) => {
                values.insert(u.id, at(0)?);
                if let Some(addr) = addresses.get(&u.src[0].id).cloned() {
                    addresses.insert(u.id, addr);
                }
                continue;
            }
            (Ops::Index, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                addresses.insert(u.id, (at(0)?, at(1)?, at(2)?, dtype));
                values.insert(
                    u.id,
                    format!("(({}*){})+({})", ctype(dtype), at(0)?, at(1)?),
                );
                continue;
            }
            (Ops::Load, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                if let Some((base, index, gate, _)) = addresses.get(&u.src[0].id) {
                    format!(
                        "({gate} ? (({}*){base})[{index}] : ({})0)",
                        ctype(dtype),
                        ctype(dtype)
                    )
                } else {
                    format!("(({}*){})[0]", ctype(dtype), at(0)?)
                }
            }
            (Ops::Store, _) => {
                let value = at(1)?;
                if status_slot.is_some_and(
                    |slot| matches!(u.src[0].arg, Arg::Slot { index,.. } if index == slot),
                ) {
                    text.push_str(&format!(
                        "{indent}if (!({value})) {{ ((int*){})[0]=1; return; }}\n",
                        at(0)?
                    ));
                    values.insert(u.id, "0".into());
                    continue;
                }
                if let Some((base, index, gate, dtype)) = addresses.get(&u.src[0].id) {
                    text.push_str(&format!(
                        "{indent}if ({gate}) (({}*){base})[{index}]={value};\n",
                        ctype(*dtype)
                    ));
                } else {
                    let dtype = if let Arg::DType(dt) = u.arg {
                        dt
                    } else {
                        u.src[1].dtype()
                    };
                    text.push_str(&format!(
                        "{indent}(({}*){})[0]={value};\n",
                        ctype(dtype),
                        at(0)?
                    ));
                }
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Range, _) => {
                value_scopes.push(values.clone());
                address_scopes.push(addresses.clone());
                let name = format!("r{}", u.id);
                text.push_str(&format!(
                    "{indent}for (long long {name}={}; {name}<{}; {name}++) {{\n",
                    at(0)?,
                    at(1)?
                ));
                values.insert(u.id, name);
                depth += 1;
                continue;
            }
            (Ops::End, _) => {
                if depth == 1 {
                    return Err("extra END".into());
                }
                depth -= 1;
                text.push_str(&format!("{}}}\n", "  ".repeat(depth)));
                values = value_scopes.pop().ok_or("missing scope")?;
                addresses = address_scopes.pop().ok_or("missing address scope")?;
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Cast, Arg::DType(DType::Bool)) => format!("({}!=0)", at(0)?),
            (Ops::Cast, Arg::DType(dtype)) => {
                format!("(({})({}))", ctype(*dtype), at(0)?)
            }
            (Ops::Bitcast, Arg::DType(dt)) if u.src[0].dtype() == *dt => at(0)?,
            (Ops::Bitcast, Arg::DType(DType::I32)) => {
                format!("chibi_int(chibi_bits({}))", at(0)?)
            }
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("chibi_float((unsigned int)({}))", at(0)?)
            }
            (Ops::Shl, _) => {
                if u.dtype() == DType::I32 {
                    format!("chibi_int((unsigned int)({}) << ({}))", at(0)?, at(1)?)
                } else {
                    format!(
                        "((long long)((unsigned long long)({}) << ({})))",
                        at(0)?,
                        at(1)?
                    )
                }
            }
            (Ops::Shr, _) => {
                format!(
                    "{}({},{})",
                    if u.dtype() == DType::I32 {
                        "chibi_shr"
                    } else {
                        "chibi_shr64"
                    },
                    at(0)?,
                    at(1)?
                )
            }
            (Ops::Neg, _) => {
                if u.dtype() == DType::I32 {
                    format!("((int)(0u-(unsigned int)({})))", at(0)?)
                } else {
                    format!("(-({}))", at(0)?)
                }
            }
            (Ops::Exp2, _) => format!("exp2f({})", at(0)?),
            (Ops::Log2, _) => format!("log2f({})", at(0)?),
            (Ops::Sin, _) => format!("sinf({})", at(0)?),
            (Ops::Sqrt, _) => format!("sqrtf({})", at(0)?),
            (Ops::Where, _) => format!("({}?{}:{})", at(0)?, at(1)?, at(2)?),
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!("chibi_max({},{})", at(0)?, at(1)?)
            }
            (Ops::Max, _) => format!("({}>{}?{}:{})", at(0)?, at(1)?, at(0)?, at(1)?),
            (Ops::Div, _) if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", at(0)?, at(1)?)
            }
            (Ops::Mod, _) => format!("chibi_mod({},{})", at(0)?, at(1)?),
            (op, _) if is_binary(*op) => {
                let symbol = match op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&",
                    Ops::Or => "|",
                    Ops::Xor => "^",
                    _ => return Err("unsupported ALU".into()),
                };
                if u.dtype() == DType::I32 && matches!(op, Ops::Add | Ops::Sub | Ops::Mul) {
                    format!(
                        "((int)((unsigned int)({}) {symbol} (unsigned int)({})))",
                        at(0)?,
                        at(1)?
                    )
                } else {
                    format!("({} {symbol} {})", at(0)?, at(1)?)
                }
            }
            (Ops::Call, _) => {
                let body = u.src.first().ok_or("CALL body missing")?;
                let Arg::Function { name, .. } = &body.arg else {
                    return Err("scalar CALL must target CUSTOM_FUNCTION".into());
                };
                format!(
                    "{}({})",
                    name,
                    u.src[1..]
                        .iter()
                        .map(get)
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            (Ops::CustomFunction, Arg::Function { name, .. }) => {
                if u.src.is_empty() {
                    values.insert(u.id, name.clone());
                    continue;
                }
                format!(
                    "{}({})",
                    name,
                    u.src.iter().map(get).collect::<Result<Vec<_>>>()?.join(",")
                )
            }
            (Ops::Sink, _) => continue,
            _ => return Err(format!("C renderer does not support {:?}", u.op)),
        };
        if u.dtype() == DType::Void {
            text.push_str(&format!("{indent}{expression};\n"));
            values.insert(u.id, "0".into());
        } else {
            let name = format!("v{}_{}", u.id, occurrence);
            occurrence += 1;
            text.push_str(&format!(
                "{indent}{} {name}={expression};\n",
                ctype(u.dtype())
            ));
            values.insert(u.id, name);
        }
    }
    if depth != 1 {
        return Err("unclosed C loop".into());
    }
    text.push_str("}\n");
    Ok(text)
}
```

`src/gpu_tensor.rs`を保存する。

```rust
// file="src/gpu_tensor.rs" mode="write"
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backends::metal::MetalDevice;
#[cfg(feature = "opencl")]
use crate::backends::opencl::OpenClDevice;
use crate::{
    Arg, Axis, Buffer, Ops, Result, Tensor, UOp,
    codegen::{rangeify, signature},
    runtime::InputSpec,
    schedule::{Schedule, create_schedule},
    uop::numel,
};
use crate::{
    gpu::{Kernel, Launch},
    queue::{CompiledQueue, HWQueue},
    renderers::{Dialect, render},
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    OpenCl,
}
enum Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDevice),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClDevice),
}
#[derive(Clone)]
enum Allocation {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalBuffer),
    #[cfg(feature = "opencl")]
    OpenCl(crate::backends::opencl::OpenClBuffer),
}
impl Device {
    fn allocate(&self, bytes: usize) -> Result<Allocation> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(d) => d.allocate(bytes.max(4)).map(Allocation::Metal),
            #[cfg(feature = "opencl")]
            Self::OpenCl(d) => d.allocate(bytes.max(4)).map(Allocation::OpenCl),
        }
    }
}
impl Allocation {
    fn retain(&self, queue: &mut HWQueue) -> UOp {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => queue.retain_metal_buffer(b.clone()),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => queue.retain_opencl_buffer(b.clone()),
        }
    }
    fn transfer(&self, queue: &mut HWQueue, pointer: UOp, bytes: UOp, read: bool) {
        let buffer = self.retain(queue);
        match (self, read) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), false) => queue.metal_write(buffer, pointer, bytes),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Self::Metal(_), true) => queue.metal_read(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), false) => queue.opencl_write(buffer, pointer, bytes),
            #[cfg(feature = "opencl")]
            (Self::OpenCl(_), true) => queue.opencl_read(buffer, pointer, bytes),
        }
    }
}
enum HostBuffer {
    Input(usize),
    Captured(usize),
    Output(usize),
    Error,
}
pub struct GpuExecutable {
    inputs: Vec<InputSpec>,
    schedule: Schedule,
    programs: Vec<UOp>,
    queue: CompiledQueue,
    input_nodes: Vec<UOp>,
    captured: Vec<Buffer>,
    transfers: Vec<HostBuffer>,
    sequence: i64,
}

fn transfer(
    queue: &mut HWQueue,
    transfers: &mut Vec<HostBuffer>,
    buffer: &Allocation,
    host: HostBuffer,
    read: bool,
) {
    let index = 2 + transfers.len() * 2;
    transfers.push(host);
    buffer.transfer(
        queue,
        HWQueue::runtime_slot(index),
        HWQueue::runtime_slot(index + 1),
        read,
    );
}

fn parallelize(linear: UOp, single_output: bool) -> (UOp, Launch) {
    let mut launch = Launch {
        groups: [1; 3],
        local: [1; 3],
    };
    let ranges = linear
        .src
        .iter()
        .filter(|u| {
            matches!(
                u.arg,
                Arg::Range {
                    axis: Axis::Loop,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let checked_indices = linear
        .src
        .iter()
        .any(|u| matches!(&u.arg, Arg::Buffer { name, .. } if name == "__chibi_error"));
    if !single_output || ranges.len() != 1 || checked_indices {
        return (linear, launch);
    }
    let range = ranges[0];
    let Arg::Int(extent) = range.src[1].arg else {
        return (linear, launch);
    };
    if extent <= 1 {
        return (linear, launch);
    }
    launch.groups[0] = extent as usize;
    let range_id = range.id;
    let mut replacements = HashMap::new();
    let mut ops = vec![];
    for u in &linear.src {
        let sources = u
            .src
            .iter()
            .map(|s| replacements.get(&s.id).unwrap_or(s).clone())
            .collect();
        let value = if u.id == range_id {
            UOp::new(
                Ops::Range,
                sources,
                Arg::Range {
                    id: 0,
                    axis: Axis::Global,
                },
            )
        } else {
            u.replace(sources)
        };
        replacements.insert(u.id, value.clone());
        ops.push(value);
    }
    (UOp::linear(ops), launch)
}

impl GpuExecutable {
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn programs(&self) -> &[UOp] {
        &self.programs
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    pub fn compile(outputs: &[Tensor], backend: Backend) -> Result<Self> {
        let roots = outputs.iter().map(|t| t.uop.clone()).collect::<Vec<_>>();
        let schedule = create_schedule(&roots)?;
        let device = match backend {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Device::Metal(MetalDevice::open()?),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            Backend::Metal => return Err("Metal requires macOS".into()),
            #[cfg(not(feature = "opencl"))]
            Backend::OpenCl => return Err("OpenCL feature is disabled".into()),
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::OpenCl(OpenClDevice::open(0)?),
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
            Backend::OpenCl => Dialect::OpenCl,
        };
        let mut queue = HWQueue::new();
        let mut transfers = vec![];
        let mut allocations = HashMap::new();
        let mut sizes = vec![];
        for life in &schedule.lifetimes {
            if sizes.len() <= life.slot {
                sizes.resize(life.slot + 1, 0)
            }
            sizes[life.slot] = sizes[life.slot].max(life.elements);
        }
        let slots = sizes
            .iter()
            .map(|&n| device.allocate(n * 4))
            .collect::<Result<Vec<_>>>()?;
        for life in &schedule.lifetimes {
            allocations.insert(life.buffer.id, slots[life.slot].clone());
        }
        let produced = schedule
            .lifetimes
            .iter()
            .map(|l| l.buffer.id)
            .collect::<HashSet<_>>();
        let mut inputs = vec![];
        let mut input_nodes = vec![];
        let mut captured = vec![];
        for node in schedule
            .buffers
            .iter()
            .filter(|u| !produced.contains(&u.id))
        {
            let Arg::Buffer { name, shape, data } = &node.arg else {
                return Err("invalid input buffer".into());
            };
            let allocation = device.allocate(numel(shape) * 4)?;
            let host = if let Some(bits) = data {
                let i = captured.len();
                captured.push(Buffer::new(
                    shape.clone(),
                    bits.iter().map(|&x| f32::from_bits(x)).collect(),
                )?);
                HostBuffer::Captured(i)
            } else {
                let i = inputs.len();
                inputs.push(InputSpec {
                    name: name.clone(),
                    shape: shape.clone(),
                });
                input_nodes.push(node.clone());
                HostBuffer::Input(i)
            };
            transfer(&mut queue, &mut transfers, &allocation, host, false);
            allocations.insert(node.id, allocation);
        }
        let error = device.allocate(4)?;
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, false);
        let mut programs = vec![];
        let mut kernels: Vec<(Kernel, Vec<Allocation>)> = vec![];
        for (i, plan) in schedule.kernels.iter().enumerate() {
            let (linear, launch) = parallelize(rangeify(&plan.sink)?, plan.writes.len() == 1);
            let sig = signature(&linear)?;
            let kernel = render(&linear, &format!("chibi_gpu_{i}"), dialect, launch)?;
            let mut bindings = vec![];
            for (slot, node) in sig.buffers {
                if bindings.len() != slot {
                    return Err("GPU buffer signature has a gap".into());
                }
                let buffer = if matches!(&node.arg, Arg::Buffer { name, .. } if name == "__chibi_error")
                {
                    error.clone()
                } else {
                    allocations
                        .get(&node.id)
                        .ok_or("missing GPU allocation")?
                        .clone()
                };
                bindings.push(buffer);
            }
            programs.push(UOp::new(
                Ops::Program,
                vec![
                    plan.sink.clone(),
                    linear,
                    UOp::new(Ops::Source, vec![], Arg::Text(kernel.source.clone())),
                ],
                Arg::Text(kernel.entry.clone()),
            ));
            kernels.push((kernel, bindings));
        }
        match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(device) => {
                let commands = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let buffers = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::Metal(b) => b.clone(),
                                _ => unreachable!(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let command = queue.retain_metal_command(device.command_buffer(&commands)?);
                for (index, (_, bindings)) in kernels.iter().enumerate() {
                    for (argument, buffer) in bindings.iter().enumerate() {
                        let buffer = buffer.retain(&mut queue);
                        queue.metal_patch(
                            command.clone(),
                            UOp::int(index as i64),
                            UOp::int(argument as i64),
                            buffer,
                            UOp::int(0),
                        );
                    }
                }
                queue.metal_submit(
                    command.clone(),
                    HWQueue::runtime_slot(0),
                    HWQueue::runtime_slot(1),
                );
                queue.metal_wait(command);
            }
            #[cfg(feature = "opencl")]
            Device::OpenCl(device) => {
                for (kernel, bindings) in &kernels {
                    let args = bindings
                        .iter()
                        .map(|b| match b {
                            Allocation::OpenCl(b) => {
                                crate::backends::opencl::Argument::Buffer(b.clone())
                            }
                            _ => unreachable!(),
                        })
                        .collect();
                    let program = device.compile(kernel)?;
                    let command =
                        queue.retain_opencl_command(program.command(kernel.launch.clone(), args)?);
                    queue.opencl_launch(command.clone());
                    queue.opencl_finish(command);
                }
            }
        }
        for (i, output) in schedule.outputs.iter().enumerate() {
            transfer(
                &mut queue,
                &mut transfers,
                &allocations[&output.id],
                HostBuffer::Output(i),
                true,
            );
        }
        transfer(&mut queue, &mut transfers, &error, HostBuffer::Error, true);
        Ok(Self {
            inputs,
            schedule,
            programs,
            queue: queue.compile("chibi_gpu_queue")?,
            input_nodes,
            captured,
            transfers,
            sequence: 0,
        })
    }
    pub fn kernel_count(&self) -> usize {
        self.programs.len()
    }
    pub fn sources(&self) -> Vec<&str> {
        self.programs
            .iter()
            .filter_map(|p| {
                p.src.iter().find_map(|s| {
                    if let (Ops::Source, Arg::Text(text)) = (s.op, &s.arg) {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
    pub fn queue_source(&self) -> Option<&str> {
        self.queue.source()
    }
    pub fn run(&mut self, inputs: &[Buffer], _vars: &HashMap<String, i64>) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "expected {} input buffers, received {}",
                self.inputs.len(),
                inputs.len()
            ));
        }
        for ((buffer, spec), node) in inputs.iter().zip(&self.inputs).zip(&self.input_nodes) {
            let shape = node.shape()?;
            if buffer.shape != shape || buffer.data.len() != numel(&shape) {
                return Err(format!(
                    "input {} expected {:?}, received {:?}",
                    spec.name, shape, buffer.shape
                ));
            }
        }
        let mut outputs = self
            .schedule
            .outputs
            .iter()
            .map(|u| {
                let shape = u.shape()?;
                Buffer::new(shape.clone(), vec![0.; numel(&shape)])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = [0f32];
        let next = self
            .sequence
            .checked_add(1)
            .ok_or("GPU timeline overflow")?;
        let mut slots = vec![self.sequence, next];
        for host in &self.transfers {
            let (pointer, bytes) = match host {
                HostBuffer::Input(i) => {
                    (inputs[*i].data.as_ptr() as usize, inputs[*i].data.len() * 4)
                }
                HostBuffer::Captured(i) => (
                    self.captured[*i].data.as_ptr() as usize,
                    self.captured[*i].data.len() * 4,
                ),
                HostBuffer::Output(i) => (
                    outputs[*i].data.as_mut_ptr() as usize,
                    outputs[*i].data.len() * 4,
                ),
                HostBuffer::Error => (error.as_mut_ptr() as usize, 4),
            };
            slots.extend([pointer as i64, bytes as i64]);
        }
        self.sequence = next;
        unsafe { self.queue.run(&mut slots) }?;
        if error[0] != 0. {
            return Err("gather/scatter index must be finite, integral, and within bounds".into());
        }
        Ok(outputs)
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub mod queue;
```

`src/queue.rs`を保存する。

```rust
// file="src/queue.rs" mode="write"
use std::collections::HashMap;
thread_local! {static ERROR:std::cell::RefCell<Option<String>>=const {std::cell::RefCell::new(None)};}
pub fn take_error() -> Option<String> {
    ERROR.with(|error| error.borrow_mut().take())
}
fn status(result: Result<(), String>) -> i64 {
    match result {
        Ok(()) => 0,
        Err(message) => {
            ERROR.with(|error| {
                let mut error = error.borrow_mut();
                if error.is_none() {
                    *error = Some(message)
                }
            });
            -1
        }
    }
}
/// Symbol addresses are supplied to the CPU ELF loader. Queues themselves are common UOp programs.
pub fn foreign_symbols() -> HashMap<String, usize> {
    let mut out = HashMap::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        out.insert(
            "chibi_metal_submit".into(),
            metal_submit as *const () as usize,
        );
        out.insert("chibi_metal_wait".into(), metal_wait as *const () as usize);
        out.insert(
            "chibi_metal_write".into(),
            metal_write as *const () as usize,
        );
        out.insert("chibi_metal_read".into(), metal_read as *const () as usize);
        out.insert(
            "chibi_metal_patch".into(),
            metal_patch as *const () as usize,
        );
    }
    #[cfg(feature = "opencl")]
    {
        out.insert("chibi_cl_launch".into(), cl_launch as *const () as usize);
        out.insert("chibi_cl_finish".into(), cl_finish as *const () as usize);
        out.insert("chibi_cl_write".into(), cl_write as *const () as usize);
        out.insert("chibi_cl_read".into(), cl_read as *const () as usize);
        out.insert(
            "chibi_cl_patch_scalar".into(),
            cl_patch_scalar as *const () as usize,
        );
    }
    out
}
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe extern "C" fn metal_submit(command: i64, wait: i64, signal: i64) -> i64 {
    let command = unsafe { &mut *(command as *mut crate::backends::metal::MetalCommandBuffer) };
    status(command.submit(wait as u64, signal as u64))
}
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe extern "C" fn metal_wait(command: i64) -> i64 {
    status(unsafe { &mut *(command as *mut crate::backends::metal::MetalCommandBuffer) }.wait())
}
#[cfg(feature = "opencl")]
unsafe extern "C" fn cl_launch(command: i64) -> i64 {
    status(unsafe { &mut *(command as *mut crate::backends::opencl::OpenClCommand) }.enqueue())
}
#[cfg(feature = "opencl")]
unsafe extern "C" fn cl_finish(command: i64) -> i64 {
    status(unsafe { &mut *(command as *mut crate::backends::opencl::OpenClCommand) }.finish())
}

use crate::{Arg, DType, Ops, UOp};
use std::any::Any;
/// A runtime program is a LINEAR of the same UOps used for kernels. Retained native objects keep callback pointers valid.
pub struct HWQueue {
    linear: UOp,
    resources: Vec<Box<dyn Any>>,
    last: Option<UOp>,
}
impl Default for HWQueue {
    fn default() -> Self {
        Self::new()
    }
}
impl HWQueue {
    pub fn new() -> Self {
        Self {
            linear: UOp::linear(vec![]),
            resources: vec![],
            last: None,
        }
    }
    pub fn linear(&self) -> &UOp {
        &self.linear
    }
    #[cfg(feature = "native")]
    pub fn compile(self, entry: &str) -> Result<CompiledQueue, String> {
        let mut required_slots = 0;
        for u in self.linear.toposort() {
            if let (Ops::Param, Arg::Slot { index, shape }) = (&u.op, &u.arg) {
                if *index != 0 {
                    return Err("runtime queue accepts one scalar-slot pointer parameter".into());
                }
                required_slots = required_slots.max(crate::uop::checked_numel(shape)?);
            }
        }
        let source = crate::codegen::render(&self.linear, entry)?;
        let program =
            crate::cpu::CpuProgram::compile_with_symbols(&source, entry, &foreign_symbols())?;
        Ok(CompiledQueue {
            queue: self,
            program,
            required_slots,
        })
    }
    fn append(&mut self, name: &str, args: Vec<UOp>) {
        let function = UOp::custom(name, DType::I64, vec![]);
        let mut sources = vec![function];
        sources.extend(args);
        let call = UOp::new(Ops::Call, sources, Arg::None);
        let call = if let Some(ref last) = self.last {
            call.after(vec![last.clone()])
        } else {
            call
        };
        self.last = Some(call.clone());
        let mut ops = self.linear.src.clone();
        let mut seen = ops
            .iter()
            .map(|u| u.id)
            .collect::<std::collections::HashSet<_>>();
        for u in call.toposort() {
            if seen.insert(u.id) {
                ops.push(u);
            }
        }
        self.linear = UOp::linear(ops);
    }
    pub fn runtime_slot(index: usize) -> UOp {
        let param = UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index: 0,
                shape: vec![index.saturating_add(1)],
            },
        );
        let ptr = UOp::new(
            Ops::Index,
            vec![param, UOp::int(index as i64), UOp::bool(true)],
            Arg::DType(DType::I64),
        );
        UOp::new(Ops::Load, vec![ptr], Arg::DType(DType::I64))
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal(
        &mut self,
        command: crate::backends::metal::MetalCommandBuffer,
        wait: UOp,
        signal: UOp,
    ) {
        let command = self.retain_metal_command(command);
        self.metal_submit(command.clone(), wait, signal);
        self.metal_wait(command);
    }
    #[cfg(feature = "opencl")]
    pub fn opencl(&mut self, command: crate::backends::opencl::OpenClCommand) {
        let command = self.retain_opencl_command(command);
        self.opencl_launch(command.clone());
        self.opencl_finish(command);
    }
}
#[cfg(feature = "native")]
pub struct CompiledQueue {
    queue: HWQueue,
    program: crate::cpu::CpuProgram,
    required_slots: usize,
}
#[cfg(feature = "native")]
impl CompiledQueue {
    pub fn linear(&self) -> &UOp {
        self.queue.linear()
    }
    /// # Safety
    /// Pointer slots and retained command handles must match this queue's UOp signature.
    /// Host buffers must remain valid with exclusive writes until all commands finish.
    pub unsafe fn run(&mut self, slots: &mut [i64]) -> Result<(), String> {
        if slots.len() < self.required_slots {
            return Err(
                "runtime scalar slots are shorter than the compiled queue signature".into(),
            );
        }
        take_error();
        let mut args = (slots.as_mut_ptr() as u64).to_ne_bytes();
        unsafe { self.program.run(&mut args) }?;
        if let Some(error) = take_error() {
            return Err(error);
        }
        Ok(())
    }
    pub fn source(&self) -> Option<&str> {
        self.program.source()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe extern "C" fn metal_patch(
    command: i64,
    index: i64,
    argument: i64,
    buffer: i64,
    offset: i64,
) -> i64 {
    if index < 0 || argument < 0 || offset < 0 {
        return status(Err("negative Metal patch argument".into()));
    }
    let command = unsafe { &mut *(command as *mut crate::backends::metal::MetalCommandBuffer) };
    let buffer = unsafe { &*(buffer as *const crate::backends::metal::MetalBuffer) };
    status(command.patch_buffer(index as usize, argument as usize, buffer, offset as usize))
}
#[cfg(feature = "opencl")]
unsafe extern "C" fn cl_patch_scalar(command: i64, index: i64, value: i64, bytes: i64) -> i64 {
    if index < 0 || !(1..=8).contains(&bytes) {
        return status(Err("invalid OpenCL scalar patch".into()));
    }
    let command = unsafe { &mut *(command as *mut crate::backends::opencl::OpenClCommand) };
    status(command.patch(
        index as usize,
        crate::backends::opencl::Argument::Scalar(value.to_ne_bytes()[..bytes as usize].to_vec()),
    ))
}
impl HWQueue {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn retain_metal_buffer(&mut self, buffer: crate::backends::metal::MetalBuffer) -> UOp {
        let buffer = Box::new(buffer);
        let ptr = (&*buffer as *const crate::backends::metal::MetalBuffer) as usize;
        self.resources.push(buffer);
        UOp::int(ptr as i64)
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn retain_metal_command(
        &mut self,
        command: crate::backends::metal::MetalCommandBuffer,
    ) -> UOp {
        let mut command = Box::new(command);
        let ptr = (&mut *command as *mut crate::backends::metal::MetalCommandBuffer) as usize;
        self.resources.push(command);
        UOp::int(ptr as i64)
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal_patch(
        &mut self,
        command: UOp,
        index: UOp,
        argument: UOp,
        buffer: UOp,
        offset: UOp,
    ) {
        self.append(
            "chibi_metal_patch",
            vec![command, index, argument, buffer, offset],
        );
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal_submit(&mut self, command: UOp, wait: UOp, signal: UOp) {
        self.append("chibi_metal_submit", vec![command, wait, signal]);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal_wait(&mut self, command: UOp) {
        self.append("chibi_metal_wait", vec![command]);
    }
    #[cfg(feature = "opencl")]
    pub fn retain_opencl_command(
        &mut self,
        command: crate::backends::opencl::OpenClCommand,
    ) -> UOp {
        let mut command = Box::new(command);
        let ptr = (&mut *command as *mut crate::backends::opencl::OpenClCommand) as usize;
        self.resources.push(command);
        UOp::int(ptr as i64)
    }
    #[cfg(feature = "opencl")]
    pub fn opencl_patch_scalar(&mut self, command: UOp, index: UOp, value: UOp, bytes: UOp) {
        self.append("chibi_cl_patch_scalar", vec![command, index, value, bytes]);
    }
    #[cfg(feature = "opencl")]
    pub fn opencl_launch(&mut self, command: UOp) {
        self.append("chibi_cl_launch", vec![command]);
    }
    #[cfg(feature = "opencl")]
    pub fn opencl_finish(&mut self, command: UOp) {
        self.append("chibi_cl_finish", vec![command]);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe extern "C" fn metal_write(buffer: i64, host: i64, bytes: i64) -> i64 {
    if bytes < 0 || (bytes > 0 && host == 0) {
        return status(Err("invalid Metal host upload span".into()));
    }
    let buffer = unsafe { &*(buffer as *const crate::backends::metal::MetalBuffer) };
    if bytes == 0 {
        return 0;
    }
    let data = unsafe { std::slice::from_raw_parts(host as *const u8, bytes as usize) };
    status(buffer.write(0, data))
}
#[cfg(all(feature = "metal", target_os = "macos"))]
unsafe extern "C" fn metal_read(buffer: i64, host: i64, bytes: i64) -> i64 {
    if bytes < 0 || (bytes > 0 && host == 0) {
        return status(Err("invalid Metal host download span".into()));
    }
    let buffer = unsafe { &*(buffer as *const crate::backends::metal::MetalBuffer) };
    let result = buffer.read(0, bytes as usize).map(|data| {
        if !data.is_empty() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host as *mut u8, data.len()) }
        }
    });
    status(result)
}
#[cfg(feature = "opencl")]
unsafe extern "C" fn cl_write(buffer: i64, host: i64, bytes: i64) -> i64 {
    if bytes < 0 || (bytes > 0 && host == 0) {
        return status(Err("invalid OpenCL host upload span".into()));
    }
    let buffer = unsafe { &*(buffer as *const crate::backends::opencl::OpenClBuffer) };
    if bytes == 0 {
        return 0;
    }
    let data = unsafe { std::slice::from_raw_parts(host as *const u8, bytes as usize) };
    status(buffer.write(0, data))
}
#[cfg(feature = "opencl")]
unsafe extern "C" fn cl_read(buffer: i64, host: i64, bytes: i64) -> i64 {
    if bytes < 0 || (bytes > 0 && host == 0) {
        return status(Err("invalid OpenCL host download span".into()));
    }
    let buffer = unsafe { &*(buffer as *const crate::backends::opencl::OpenClBuffer) };
    let result = buffer.read(0, bytes as usize).map(|data| {
        if !data.is_empty() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host as *mut u8, data.len()) }
        }
    });
    status(result)
}
impl HWQueue {
    #[cfg(feature = "opencl")]
    pub fn retain_opencl_buffer(&mut self, buffer: crate::backends::opencl::OpenClBuffer) -> UOp {
        let buffer = Box::new(buffer);
        let ptr = (&*buffer as *const crate::backends::opencl::OpenClBuffer) as usize;
        self.resources.push(buffer);
        UOp::int(ptr as i64)
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal_write(&mut self, buffer: UOp, host: UOp, bytes: UOp) {
        self.append("chibi_metal_write", vec![buffer, host, bytes]);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn metal_read(&mut self, buffer: UOp, host: UOp, bytes: UOp) {
        self.append("chibi_metal_read", vec![buffer, host, bytes]);
    }
    #[cfg(feature = "opencl")]
    pub fn opencl_write(&mut self, buffer: UOp, host: UOp, bytes: UOp) {
        self.append("chibi_cl_write", vec![buffer, host, bytes]);
    }
    #[cfg(feature = "opencl")]
    pub fn opencl_read(&mut self, buffer: UOp, host: UOp, bytes: UOp) {
        self.append("chibi_cl_read", vec![buffer, host, bytes]);
    }
}
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Cast,
        Ops::Bitcast,
        Ops::Shl,
        Ops::Shr,
        Ops::Xor,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Div,
        Ops::Mod,
        Ops::Neg,
        Ops::Exp,
        Ops::Log,
        Ops::Sin,
        Ops::Cos,
        Ops::Sqrt,
        Ops::Exp2,
        Ops::Log2,
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Reshape,
        Ops::Permute,
        Ops::Expand,
        Ops::Shrink,
        Ops::Pad,
        Ops::Flip,
        Ops::Stack,
        Ops::Gather,
        Ops::ScatterAdd,
        Ops::Reduce,
        Ops::Detach,
        Ops::Stage,
        Ops::Range,
        Ops::Index,
        Ops::Load,
        Ops::Store,
        Ops::End,
        Ops::After,
        Ops::Call,
        Ops::Sink,
        Ops::Linear,
        Ops::Program,
        Ops::Source,
        Ops::Binary,
        Ops::CustomFunction,
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uir.rs` の `fn encode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn encode_arg"
fn encode_arg(a: &Arg) -> String {
    match a {
        Arg::None => "none".into(),
        Arg::Float(v) => format!("float;{:016x}", v.value().to_bits()),
        Arg::I32(v) => format!("i32;{v}"),
        Arg::Int(v) => format!("i64;{v}"),
        Arg::Bool(v) => format!("bool;{}", *v as u8),
        Arg::Shape(v) => format!("shape;{}", numbers(v)),
        Arg::Axes(v) => format!("axes;{}", numbers(v)),
        Arg::Pairs(v) => {
            format!(
                "pairs;{}",
                numbers(&v.iter().flat_map(|(a, b)| [a, b]).collect::<Vec<_>>())
            )
        }
        Arg::Axis(v) => format!("axis;{v}"),
        Arg::Reduction { op, axes, keepdim } => {
            format!("reduction;{op:?};{};{}", numbers(axes), *keepdim as u8)
        }
        Arg::Buffer { name, shape, data } => {
            format!(
                "buffer;{};{};{}",
                hex(name.as_bytes()),
                numbers(shape),
                data.as_ref().map(|v| numbers(v)).unwrap_or("-".into())
            )
        }
        Arg::Range { id, axis } => format!("range;{id};{axis:?}"),
        Arg::Allocation { id, dtype, size } => {
            format!("allocation;{id};{dtype:?};{size}")
        }
        Arg::Id(v) => format!("id;{v}"),
        Arg::DType(dt) => format!("dtype;{dt:?}"),
        Arg::Text(s) => format!("text;{}", hex(s.as_bytes())),
        Arg::Bytes(v) => format!("bytes;{}", hex(v)),
        Arg::Slot { index, shape } => format!("slot;{index};{}", numbers(shape)),
        Arg::ScalarParam { index, dtype } => format!("scalar;{index};{dtype:?}"),
        Arg::Function { name, dtype } => format!("function;{};{dtype:?}",hex(name.as_bytes())),

    }
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["float", v] if v.len() == 16 && v.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Arg::Float(crate::uop::ConstFloat::new(f64::from_bits(
                u64::from_str_radix(v, 16).map_err(|_| "invalid float bits")?
            )))
        }
        ["i32", v] => Arg::I32(number(v)?),
        ["i64", v] => Arg::Int(number(v)?),
        ["bool", v] => Arg::Bool(boolean(v)?),
        ["shape", v] => Arg::Shape(list(v)?),
        ["axes", v] => Arg::Axes(list(v)?),
        ["pairs", v] => {
            let v: Vec<usize> = list(v)?;
            if v.len() % 2 != 0 {
                return Err("odd pairs list".into());
            }
            Arg::Pairs(v.chunks_exact(2).map(|p| (p[0], p[1])).collect())
        }
        ["axis", v] => Arg::Axis(number(v)?),
        ["reduction", o, v, k] => Arg::Reduction {
            op: op(o)?,
            axes: list(v)?,
            keepdim: boolean(k)?,
        },
        ["buffer", n, v, d] => Arg::Buffer {
            name: string(n)?,
            shape: list(v)?,
            data: if *d == "-" { None } else { Some(list(d)?) },
        },
        ["range", id, a] => Arg::Range {
            id: number(id)?,
            axis: match *a {
                "Loop" => Axis::Loop,
                        "Global" => Axis::Global,
                "Reduce" => Axis::Reduce,
                _ => return Err("unknown axis".into()),
            },
        },
        ["allocation", id, dt, size] => Arg::Allocation {
            id: number(id)?,
            dtype: dtype(dt)?,
            size: number(size)?,
        },
        ["id", v] => Arg::Id(number(v)?),
        ["dtype", dt] => Arg::DType(dtype(dt)?),
        ["text", v] => Arg::Text(string(v)?),
        ["bytes", v] => Arg::Bytes(unhex(v)?),
        ["slot", i, v] => Arg::Slot {
            index: number(i)?,
            shape: list(v)?,
        },
        ["function",name,dt] => Arg::Function{name:string(name)?,dtype:dtype(dt)?},
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
```

`src/uir.rs` の `fn validate_node`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn validate_node"
fn validate_node(u: &UOp) -> Result<()> {
    if crate::uop::is_binary(u.op) || crate::uop::is_unary(u.op) || u.op == Ops::Where {
        if u.src.iter().all(|s| {
            matches!(
                s.dtype(),
                DType::Bool | DType::I32 | DType::I64 | DType::F32
            ) && s.shape().is_ok_and(|shape| shape.is_empty())
        }) {
            crate::scalar::verify_node(u)?;
        } else {
            u.shape()?;
        }
    }
    match (&u.op, &u.arg) {
        (Ops::Const, _) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::ScalarParam { .. }) => {
            crate::scalar::verify_node(u)?;
        }
        (Ops::Param, Arg::Slot { .. })
        | (Ops::Buffer, Arg::Buffer { .. })
        | (Ops::Alloc, Arg::Allocation { .. })
            if u.src.is_empty() =>
        {
            if u.op == Ops::Buffer {
                u.shape()?;
            }
        }
        (Ops::Param | Ops::Buffer | Ops::Alloc, _) => {
            return Err("invalid leaf metadata".into());
        }
        (Ops::Range, Arg::Range { .. })
            if u.src.len() == 2 && u.src.iter().all(|s| s.dtype() == DType::I64) => {}
        (Ops::Range, _) => return Err("invalid RANGE".into()),
        (Ops::End, Arg::None)
            if !u.src.is_empty() && u.src.last().is_some_and(|r| r.op == Ops::Range) => {}
        (Ops::End, _) => return Err("invalid END".into()),
        (Ops::Source, Arg::Text(_))
        | (Ops::Binary, Arg::Bytes(_))
        | (Ops::Invalid, Arg::Text(_))
            if u.src.is_empty() => {}
        (Ops::Source | Ops::Binary | Ops::Invalid, _) => {
            return Err("invalid source payload".into());
        }
        (Ops::Load, Arg::None | Arg::DType(_))
            if u.src.len() == 1 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Store, Arg::None | Arg::DType(_))
            if u.src.len() == 2 && u.src[0].dtype() == DType::Ptr => {}
        (Ops::Load | Ops::Store, _) => return Err("invalid memory operation".into()),
        (Ops::Linear, Arg::None) => {
            crate::codegen::verify_lowered(u)?;
            validate_linear_scope(u)?;
        }
        (Ops::Linear, _) => return Err("invalid LINEAR".into()),
        (Ops::Call, Arg::Id(count))
            if u.src.first().is_some_and(|b| b.op == Ops::Stack)
                && u.src.len().checked_sub(1) == Some(*count) => {}
        (Ops::Call, Arg::None) if u.src.first().is_some_and(|b| b.op==Ops::CustomFunction) => {},
        (Ops::CustomFunction, Arg::Function{..}) => {},
        (Ops::Call, _) => {
            return Err("invalid CALL".into());
        }
        _ => {}
    }
    Ok(())
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Invalid,
    Const,
    Buffer,
    Alloc,
    Param,
    Cast,
    Bitcast,
    Shl,
    Shr,
    Xor,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Exp2,
    Log2,
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Reshape,
    Permute,
    Expand,
    Shrink,
    Pad,
    Flip,
    Stack,
    Gather,
    ScatterAdd,
    Reduce,
    Detach,
    Stage,
    Range,
    Index,
    Load,
    Store,
    End,
    After,
    Call,
    Sink,
    Linear,
    Program,
    Source,
    Binary,
    CustomFunction,
}
```

`src/uop.rs` の `pub enum Arg`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Arg"
pub enum Arg {
    None,
    Float(ConstFloat),
    I32(i32),
    Int(i64),
    Bool(bool),
    Shape(Vec<usize>),
    Axes(Vec<usize>),
    Pairs(Vec<(usize, usize)>),
    Axis(usize),
    Reduction {
        op: Ops,
        axes: Vec<usize>,
        keepdim: bool,
    },
    Buffer {
        name: String,
        shape: Vec<usize>,
        data: Option<Vec<u32>>,
    },
    Range {
        id: usize,
        axis: Axis,
    },
    Allocation {
        id: usize,
        dtype: DType,
        size: usize,
    },
    Id(usize),
    DType(DType),
    Text(String),
    Bytes(Vec<u8>),
    Slot {
        index: usize,
        shape: Vec<usize>,
    },
    ScalarParam {
        index: usize,
        dtype: DType,
    },
    Function { name: String, dtype: DType },
}
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_dtype`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_dtype"
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::Float(_) => DType::F32,
                Arg::I32(_) => DType::I32,
                Arg::Int(_) => DType::I64,
                Arg::Bool(_) => DType::Bool,
                _ => DType::Void,
            },
            Ops::Param => match self.arg {
                Arg::ScalarParam { dtype, .. } => dtype,
                _ => DType::Ptr,
            },
            Ops::Buffer | Ops::Alloc => DType::Ptr,
            Ops::Index => {
                if self.src.first().is_some_and(|s| s.op == Ops::Call) {
                    self.src[0].dtype()
                } else {
                    DType::Ptr
                }
            }
            Ops::Range => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::CustomFunction => {
                if let Arg::Function { dtype, .. } = self.arg { dtype } else { DType::Void }
            }
            Ops::Load => match self.arg {
                Arg::DType(dt) => dt,
                _ => DType::F32,
            },
            Ops::Gather | Ops::ScatterAdd => DType::F32,
            Ops::Call => self.src.first().map_or(DType::Void, |x| x.dtype()),
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Store
            | Ops::End
            | Ops::Sink
            | Ops::Linear
            | Ops::Program
            | Ops::Source
            | Ops::Binary
            | Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| {
                if x.dtype() == DType::Ptr {
                    DType::F32
                } else {
                    x.dtype()
                }
            }),
        }
    }
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_shape`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_shape"
    fn infer_shape(&self) -> Result<Vec<usize>> {
        let err = || format!("invalid {:?} arguments", self.op);
        let source = || self.src.first().ok_or_else(err)?.shape();
        match (&self.op, &self.arg) {
            (Ops::Invalid, Arg::Text(s)) => Err(s.clone()),
            (Ops::Buffer, Arg::Buffer { shape, data, .. }) => {
                checked_numel(shape)?;
                if data.as_ref().is_some_and(|d| d.len() != numel(shape)) {
                    return Err("buffer data length disagrees with shape".into());
                }
                Ok(shape.clone())
            }
            (Ops::Param, Arg::Slot { .. }) => Ok(vec![]),
            (Ops::Param, Arg::ScalarParam { .. }) => Ok(vec![]),
            (Ops::Const, _) | (Ops::Range, _) | (Ops::Load, _) | (Ops::CustomFunction, _) => Ok(vec![]),
            (Ops::Reshape, Arg::Shape(shape)) => {
                if checked_numel(shape)? != checked_numel(&source()?)? {
                    return Err("reshape changes element count".into());
                }
                Ok(shape.clone())
            }
            (Ops::Expand, Arg::Shape(shape)) => {
                let old = source()?;
                if shape.len() != old.len()
                    || old.iter().zip(shape).any(|(&a, &b)| a != b && a != 1)
                {
                    return Err("invalid expand dimensions".into());
                }
                checked_numel(shape)?;
                Ok(shape.clone())
            }
            (Ops::Permute, Arg::Axes(axes)) => {
                let old = source()?;
                let set: HashSet<_> = axes.iter().copied().collect();
                if axes.len() != old.len()
                    || set.len() != old.len()
                    || axes.iter().any(|&a| a >= old.len())
                {
                    return Err("invalid permutation".into());
                }
                Ok(axes.iter().map(|&a| old[a]).collect())
            }
            (Ops::Shrink, Arg::Pairs(pairs)) => {
                let old = source()?;
                if pairs.len() != old.len()
                    || pairs.iter().zip(old).any(|(&(a, b), n)| a > b || b > n)
                {
                    return Err("shrink bounds outside shape".into());
                }
                Ok(pairs.iter().map(|&(a, b)| b - a).collect())
            }
            (Ops::Pad, Arg::Pairs(pairs)) => {
                let old = source()?;
                if pairs.len() != old.len() {
                    return Err("pad rank mismatch".into());
                }
                old.iter()
                    .zip(pairs)
                    .map(|(&n, &(a, b))| {
                        n.checked_add(a)
                            .and_then(|x| x.checked_add(b))
                            .ok_or_else(|| "pad overflow".into())
                    })
                    .collect()
            }
            (Ops::Flip, Arg::Axes(axes)) => {
                let s = source()?;
                if axes.iter().any(|&a| a >= s.len()) {
                    return Err("flip axis outside rank".into());
                }
                Ok(s)
            }
            (Ops::Reduce, Arg::Reduction { op, axes, keepdim }) => {
                let old = source()?;
                if !matches!(op, Ops::Add | Ops::Max)
                    || axes.iter().any(|&a| a >= old.len())
                    || axes.iter().copied().collect::<HashSet<_>>().len() != axes.len()
                {
                    return Err("invalid reduction axes or operation".into());
                }
                Ok(old
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &n)| {
                        if axes.contains(&i) {
                            keepdim.then_some(1)
                        } else {
                            Some(n)
                        }
                    })
                    .collect())
            }
            (Ops::Stack, Arg::Axis(axis)) => {
                let old = source()?;
                if *axis > old.len()
                    || self
                        .src
                        .iter()
                        .any(|x| x.shape().ok().as_ref() != Some(&old))
                {
                    return Err("stack shapes differ".into());
                }
                let mut s = old;
                s.insert(*axis, self.src.len());
                Ok(s)
            }
            (Ops::Gather, Arg::Axis(axis)) => {
                let s = source()?;
                let idx = self.src.get(1).ok_or_else(err)?.shape()?;
                if *axis >= s.len() {
                    return Err("gather axis outside rank".into());
                }
                let mut out = s[..*axis].to_vec();
                out.extend(idx);
                out.extend_from_slice(&s[axis + 1..]);
                Ok(out)
            }
            (Ops::ScatterAdd, Arg::Axis(axis)) => {
                let s = source()?;
                let idx = self.src.get(1).ok_or_else(err)?.shape()?;
                let values = self.src.get(2).ok_or_else(err)?.shape()?;
                if *axis >= s.len() {
                    return Err("scatter axis outside rank".into());
                }
                let mut expected = s[..*axis].to_vec();
                expected.extend(idx);
                expected.extend_from_slice(&s[axis + 1..]);
                if values != expected {
                    return Err("scatter values shape mismatch".into());
                }
                Ok(s)
            }
            (Ops::Where, _) => {
                if self.src.len() != 3 {
                    return Err(err());
                }
                let s = broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)?;
                broadcast_shape(&s, &self.src[2].shape()?)
            }
            (op, _) if is_binary(*op) => {
                if self.src.len() != 2 {
                    return Err(err());
                }
                broadcast_shape(&self.src[0].shape()?, &self.src[1].shape()?)
            }
            (op, _) if is_unary(*op) || matches!(op, Ops::Detach | Ops::Stage | Ops::After) => {
                source()
            }
            _ => Err(err()),
        }
    }
```

`src/uop.rs` の `impl UOp` 内を追記する。

```rust
// file="src/uop.rs" mode="append" within="impl UOp"
    pub fn custom(name: &str, dtype: DType, src: Vec<UOp>) -> Self {
        Self::new(
            Ops::CustomFunction,
            src,
            Arg::Function {
                name: name.into(),
                dtype,
            },
        )
    }
```

`tests/runtime_patch.rs`を保存する。

```rust
// file="tests/runtime_patch.rs" mode="write"
#![cfg(any(feature = "metal", feature = "opencl"))]
use chibigrad::cpu::CpuProgram;
use chibigrad::{UOp, codegen};
use chibigrad::{
    gpu::{Kernel, Launch},
    queue::{HWQueue, foreign_symbols, take_error},
};
static GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn launch() -> Launch {
    Launch {
        groups: [2, 1, 1],
        local: [4, 1, 1],
    }
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn floats(v: Vec<u8>) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
}
fn compile(queue: &HWQueue) -> CpuProgram {
    let text = chibigrad::uir::print(queue.linear());
    assert_eq!(
        chibigrad::uir::print(&chibigrad::uir::parse(&text).unwrap()),
        text
    );
    let src = codegen::render(queue.linear(), "runtime_patch").unwrap();
    CpuProgram::compile_with_symbols(&src, "runtime_patch", &foreign_symbols()).unwrap()
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_patch_and_timeline_are_runtime_uops() {
    let _gpu = GPU_TEST.lock().unwrap();
    use chibigrad::backends::metal::MetalDevice;
    let dev = MetalDevice::open().unwrap();
    let input = dev.allocate(64).unwrap();
    input.write(0, &bytes(&[1.; 8])).unwrap();
    input.write(32, &bytes(&[10.; 8])).unwrap();
    let out = dev.allocate(32).unwrap();
    let spec=Kernel{source:"#include <metal_stdlib>\nusing namespace metal;kernel void add(device float*a[[buffer(0)]],device float*b[[buffer(1)]],uint id[[thread_position_in_grid]]){b[id]=a[id]+1.0f;}".into(),entry:"add".into(),launch:launch(),shared_bytes:0};
    let p = dev.compile(&spec).unwrap();
    let commands = dev
        .command_buffer(&[(p, launch(), vec![input.clone(), out.clone()])])
        .unwrap();
    let mut q = HWQueue::new();
    let input = q.retain_metal_buffer(input);
    let command = q.retain_metal_command(commands);
    q.metal_patch(
        command.clone(),
        UOp::int(0),
        UOp::int(0),
        input,
        HWQueue::runtime_slot(0),
    );
    q.metal_submit(command.clone(), UOp::int(0), HWQueue::runtime_slot(1));
    q.metal_wait(command);
    let cpu = compile(&q);
    let mut slots = [0i64, 1];
    let mut args = (slots.as_mut_ptr() as u64).to_ne_bytes();
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(take_error(), None);
    assert_eq!(floats(out.read(0, 32).unwrap()), vec![2.; 8]);
    slots = [32, 2];
    std::hint::black_box(&slots);
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(take_error(), None);
    assert_eq!(floats(out.read(0, 32).unwrap()), vec![11.; 8]);
}
#[cfg(feature = "opencl")]
#[test]
fn cl_scalar_patch_is_runtime_uop() {
    let _gpu = GPU_TEST.lock().unwrap();
    use chibigrad::backends::opencl::{Argument, OpenClDevice};
    let names = match OpenClDevice::devices() {
        Ok(d) if !d.is_empty() => d,
        other => {
            eprintln!("OpenCL unavailable: {other:?}");
            return;
        }
    };
    eprintln!("OpenCL runtime patch hardware {names:?}");
    let dev = OpenClDevice::open(0).unwrap();
    let input = dev.allocate(32).unwrap();
    input.write(0, &bytes(&[1.; 8])).unwrap();
    let out = dev.allocate(32).unwrap();
    let spec=Kernel{source:"__kernel void add(__global float*a,__global float*b,float s){b[get_global_id(0)]=a[get_global_id(0)]+s;}".into(),entry:"add".into(),launch:launch(),shared_bytes:0};
    let p = dev.compile(&spec).unwrap();
    let command = p
        .command(
            launch(),
            vec![
                Argument::Buffer(input),
                Argument::Buffer(out.clone()),
                Argument::Scalar(0f32.to_ne_bytes().to_vec()),
            ],
        )
        .unwrap();
    let mut q = HWQueue::new();
    let command = q.retain_opencl_command(command);
    q.opencl_patch_scalar(
        command.clone(),
        UOp::int(2),
        HWQueue::runtime_slot(0),
        UOp::int(4),
    );
    q.opencl_launch(command.clone());
    q.opencl_finish(command);
    let cpu = compile(&q);
    let mut slots = [2f32.to_bits() as i64];
    let mut args = (slots.as_mut_ptr() as u64).to_ne_bytes();
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(take_error(), None);
    assert_eq!(floats(out.read(0, 32).unwrap()), vec![3.; 8]);
    slots[0] = 7f32.to_bits() as i64;
    std::hint::black_box(&slots);
    unsafe { cpu.run(&mut args) }.unwrap();
    assert_eq!(take_error(), None);
    assert_eq!(floats(out.read(0, 32).unwrap()), vec![8.; 8]);
}

#[cfg(all(feature = "native", feature = "metal", target_os = "macos"))]
#[test]
fn compiled_queue_owns_native_resources_and_reports_runtime_errors() {
    let _gpu = GPU_TEST.lock().unwrap();
    use chibigrad::backends::metal::MetalDevice;
    let dev = MetalDevice::open().unwrap();
    let input = dev.allocate(32).unwrap();
    input.write(0, &bytes(&[4.; 8])).unwrap();
    let out = dev.allocate(32).unwrap();
    let spec=Kernel{source:"#include <metal_stdlib>\nusing namespace metal;kernel void copy(device float*a[[buffer(0)]],device float*b[[buffer(1)]],uint id[[thread_position_in_grid]]){b[id]=a[id];}".into(),entry:"copy".into(),launch:launch(),shared_bytes:0};
    let p = dev.compile(&spec).unwrap();
    let commands = dev
        .command_buffer(&[(p, launch(), vec![input.clone(), out.clone()])])
        .unwrap();
    let mut q = HWQueue::new();
    let input_handle = q.retain_metal_buffer(input);
    let output_handle = q.retain_metal_buffer(out);
    q.metal_write(input_handle, HWQueue::runtime_slot(1), UOp::int(32));
    q.metal(commands, UOp::int(0), HWQueue::runtime_slot(0));
    q.metal_read(output_handle, HWQueue::runtime_slot(2), UOp::int(32));
    let mut compiled = q.compile("owned_runtime").unwrap();
    drop(dev);
    assert!(unsafe { compiled.run(&mut []) }.is_err());
    let mut input = [4f32; 8];
    let mut output = [0f32; 8];
    let mut slots = [1, input.as_ptr() as i64, output.as_mut_ptr() as i64];
    unsafe { compiled.run(&mut slots) }.unwrap();
    assert_eq!(output, [4.; 8]);
    assert!(
        unsafe { compiled.run(&mut slots) }
            .unwrap_err()
            .contains("timeline")
    );
    input.fill(8.);
    std::hint::black_box(&input);
    slots[0] = 2;
    unsafe { compiled.run(&mut slots) }.unwrap();
    assert_eq!(output, [8.; 8]);
}
```

```sh
# check step=17.05 requires=metal,opencl
cargo test --features metal,opencl --test runtime_patch --test tensor_gpu -- --nocapture
```

キューの引数更新、リソース寿命、エラー伝達の 3 テストと以前の GPU 6 テストが実機で通る。

## 17.06 出力ループを分割する

定数境界の出力ループを外側と内側へ分け、端数を INDEX のマスクへ伝える。ループをまたぐ状態と不明な CALL を拒否する。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod optimization;
```

`src/optimization.rs`を保存する。

```rust
// file="src/optimization.rs" mode="write"
use crate::{Arg, Axis, Ops, Result, UOp, codegen};
use std::collections::{HashMap, HashSet};

fn emit(node: UOp, out: &mut Vec<UOp>, seen: &mut HashSet<u64>) {
    if seen.insert(node.id) {
        for source in &node.src {
            emit(source.clone(), out, seen);
        }
        out.push(node);
    }
}

fn replace(node: &UOp, replacements: &mut HashMap<u64, UOp>, guard: Option<&UOp>) -> UOp {
    if let Some(replaced) = replacements.get(&node.id) {
        return replaced.clone();
    }
    let mut sources = node
        .src
        .iter()
        .map(|s| replace(s, replacements, guard))
        .collect::<Vec<_>>();
    if node.op == Ops::Index
        && let Some(guard) = guard
    {
        sources[2] = sources[2].and(guard);
    }
    let result = UOp::new(node.op, sources, node.arg.clone());
    replacements.insert(node.id, result.clone());
    result
}

pub fn split_range(linear: &UOp, axis: usize, factor: usize) -> Result<UOp> {
    let linear = complete(linear)?;
    if factor == 0 || factor > i64::MAX as usize {
        return Err("invalid split factor".into());
    }
    let range = linear
        .src
        .iter()
        .find(|u| matches!(u.arg,Arg::Range{id,..} if id == axis) && u.op == Ops::Range)
        .ok_or("unknown range axis")?;
    let Arg::Range { axis: kind, .. } = range.arg else {
        unreachable!()
    };
    if kind != Axis::Loop {
        return Err("split_range expects a serial output axis".into());
    }
    let (begin, finish) = range_span(&linear, axis)?;
    let body = &linear.src[begin + 1..finish];
    for node in body {
        if matches!(node.op, Ops::Call) {
            return Err("unknown effects prevent splitting".into());
        }
        if node.op == Ops::Store {
            let base = memory_base(&node.src[0]);
            if base.op == Ops::Alloc {
                if !body.iter().any(|u| u == base) {
                    return Err("loop-carried accumulator prevents guarded splitting".into());
                }
            } else if node.src[0].op != Ops::Index {
                return Err("split needs indexed external stores".into());
            }
        }
    }
    let next_id = linear
        .src
        .iter()
        .filter_map(|u| match u.arg {
            Arg::Range { id, .. } => Some(id),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("split axis overflow")?;
    next_id.checked_add(1).ok_or("split axis overflow")?;
    let start = constant(&range.src[0])?;
    let end = constant(&range.src[1])?;
    if end < start {
        return Err("negative loop count".into());
    }
    end.checked_sub(start)
        .and_then(|count| count.checked_add(factor as i64 - 1))
        .ok_or("split count overflow")?;
    let width = UOp::int(factor as i64);
    let count = range.src[1].sub(&range.src[0]);
    let outer = UOp::range(
        count.add(&width.sub(&UOp::int(1))).div(&width),
        next_id,
        Axis::Loop,
    );
    let inner = UOp::range(width.clone(), next_id + 1, Axis::Loop);
    let index = range.src[0].add(&outer.mul(&width)).add(&inner);
    let guard = index.lt(&range.src[1]);
    let mut replacements = HashMap::<u64, UOp>::new();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut inside = false;
    for node in &linear.src {
        if node == range {
            emit(outer.clone(), &mut out, &mut seen);
            emit(inner.clone(), &mut out, &mut seen);
            emit(index.clone(), &mut out, &mut seen);
            replacements.insert(node.id, index.clone());
            inside = true;
            continue;
        }
        if node.op == Ops::End && node.src.last() == Some(range) {
            let body = node
                .src
                .first()
                .and_then(|s| replacements.get(&s.id))
                .cloned()
                .unwrap_or_else(|| inner.clone());
            let end = UOp::new(Ops::End, vec![body, inner.clone()], Arg::None);
            out.push(end.clone());
            seen.insert(end.id);
            let outer_end = UOp::new(Ops::End, vec![end, outer.clone()], Arg::None);
            out.push(outer_end.clone());
            seen.insert(outer_end.id);
            replacements.insert(node.id, outer_end);
            inside = false;
            continue;
        }
        let rewritten = replace(node, &mut replacements, inside.then_some(&guard));
        if node.op == Ops::End {
            out.push(rewritten.clone());
            seen.insert(rewritten.id);
        } else {
            emit(rewritten, &mut out, &mut seen);
        }
    }
    let result = UOp::linear(out);
    codegen::verify_lowered(&result)?;
    Ok(result)
}

pub(crate) fn complete(linear: &UOp) -> Result<UOp> {
    codegen::verify_lowered(linear)?;
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for s in u.ordered_sources() {
            add(s, out, seen, effects, scopes);
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End
            && let Some(parent) = scopes.pop()
        {
            *seen = parent;
        }
    }
    let mut out = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    for u in &linear.src {
        add(u, &mut out, &mut seen, &mut effects, &mut scopes);
    }
    let result = UOp::linear(out);
    codegen::verify_lowered(&result)?;
    Ok(result)
}

fn range_span(linear: &UOp, axis: usize) -> Result<(usize, usize)> {
    let start = linear
        .src
        .iter()
        .position(|u| u.op == Ops::Range && matches!(u.arg, Arg::Range { id, .. } if id == axis))
        .ok_or("unknown range axis")?;
    let end = linear
        .src
        .iter()
        .position(|u| u.op == Ops::End && u.src.last() == Some(&linear.src[start]))
        .ok_or("missing range END")?;
    Ok((start, end))
}

fn constant(u: &UOp) -> Result<i64> {
    if let Arg::Int(value) = u.arg {
        return Ok(value);
    }
    let a = || constant(&u.src[0]);
    let b = || constant(&u.src[1]);
    let value = match u.op {
        Ops::Add => a()?.checked_add(b()?),
        Ops::Sub => a()?.checked_sub(b()?),
        Ops::Mul => a()?.checked_mul(b()?),
        Ops::Div => a()?.checked_div_euclid(b()?),
        Ops::Mod => a()?.checked_rem_euclid(b()?),
        _ => None,
    };
    value.ok_or_else(|| "optimization requires representable constant loop bounds".into())
}

fn memory_base(u: &UOp) -> &UOp {
    if matches!(u.op, Ops::Index | Ops::After) {
        memory_base(&u.src[0])
    } else {
        u
    }
}
```

`tests/loop_optimization.rs`を保存する。

```rust
// file="tests/loop_optimization.rs" mode="write"
use chibigrad::{Arg, Axis, Ops, UOp, codegen, cpu::CpuProgram, optimization::split_range};

#[test]
fn split_range_keeps_a_partial_tail_guarded() {
    let input = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 0,
            shape: vec![17],
        },
    );
    let output = UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index: 1,
            shape: vec![17],
        },
    );
    let r = UOp::range(UOp::int(17), 0, Axis::Loop);
    let value = input
        .index(r.clone(), UOp::bool(true))
        .load()
        .mul(&UOp::f32(2.));
    let store = output.index(r.clone(), UOp::bool(true)).store(value);
    let end = UOp::new(Ops::End, vec![store.clone(), r.clone()], Arg::None);
    let linear = UOp::linear(vec![r, store, end]);
    let optimized = split_range(&linear, 0, 4).unwrap();
    assert_eq!(
        optimized.src.iter().filter(|u| u.op == Ops::Range).count(),
        2
    );
    let source = codegen::render(&optimized, "split_entry").unwrap();
    let native = CpuProgram::compile(&source, "split_entry").unwrap();
    let input = (0..17).map(|x| x as f32).collect::<Vec<_>>();
    let mut output = vec![0.0f32; 18];
    output[17] = 12345.;
    let mut args = (input.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { native.run(&mut args) }.unwrap();
    assert_eq!(
        &output[..17],
        &input.iter().map(|v| v * 2.).collect::<Vec<_>>()
    );
    assert_eq!(output[17], 12345.);
    assert!(split_range(&linear, 0, 0).is_err());
}
```

```sh
# check step=17.06
cargo test --test loop_optimization
```

17 要素を幅 4 で分割しても全結果が一致し、18 番目のカナリアは変わらない。

## 17.07 独立した出力ループを交換する

完全なループ入れ子とアフィンな出力添字を調べ、独立性を証明できる軸だけを交換する。依存する境界や重なる出力は拒否する。

`src/optimization.rs`を追記する。

```rust
// file="src/optimization.rs" mode="append"


fn rebuild(u: &UOp, map: &mut HashMap<u64, UOp>) -> UOp {
    if let Some(v) = map.get(&u.id) {
        return v.clone();
    }
    let sources = u.src.iter().map(|s| rebuild(s, map)).collect();
    let result = UOp::new(u.op, sources, u.arg.clone());
    map.insert(u.id, result.clone());
    result
}


fn affine(u: &UOp, ranges: &[UOp]) -> Option<(i64, Vec<i64>)> {
    if let Some(i) = ranges.iter().position(|r| r == u) {
        let mut coefficients = vec![0; ranges.len()];
        coefficients[i] = 1;
        return Some((0, coefficients));
    }
    match u.op {
        Ops::Const => match u.arg {
            Arg::Int(n) => Some((n, vec![0; ranges.len()])),
            _ => None,
        },
        Ops::Add | Ops::Sub => {
            let (a, av) = affine(&u.src[0], ranges)?;
            let (b, bv) = affine(&u.src[1], ranges)?;
            let sign = if u.op == Ops::Add { 1i64 } else { -1i64 };
            Some((
                a.checked_add(b.checked_mul(sign)?)?,
                av.into_iter()
                    .zip(bv)
                    .map(|(x, y)| x.checked_add(y.checked_mul(sign)?))
                    .collect::<Option<Vec<_>>>()?,
            ))
        }
        Ops::Mul => {
            let (a, av) = affine(&u.src[0], ranges)?;
            let (b, bv) = affine(&u.src[1], ranges)?;
            if av.iter().all(|&x| x == 0) {
                Some((
                    a.checked_mul(b)?,
                    bv.into_iter()
                        .map(|x| x.checked_mul(a))
                        .collect::<Option<Vec<_>>>()?,
                ))
            } else if bv.iter().all(|&x| x == 0) {
                Some((
                    a.checked_mul(b)?,
                    av.into_iter()
                        .map(|x| x.checked_mul(b))
                        .collect::<Option<Vec<_>>>()?,
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}


fn injective(index: &UOp, ranges: &[UOp]) -> Result<()> {
    let (_, coefficients) =
        affine(index, ranges).ok_or("cannot prove independent affine output addresses")?;
    let mut strides = vec![];
    for (r, coefficient) in ranges.iter().zip(coefficients) {
        let count = constant(&r.src[1])?
            .checked_sub(constant(&r.src[0])?)
            .ok_or("loop count overflow")?;
        if count < 0 {
            return Err("negative loop count".into());
        }
        if count > 1 {
            if coefficient == 0 {
                return Err("output address aliases across loop iterations".into());
            }
            strides.push((coefficient.checked_abs().ok_or("stride overflow")?, count));
        }
    }
    strides.sort_unstable();
    let mut span = 0i64;
    for (stride, count) in strides {
        if stride <= span {
            return Err("cannot prove disjoint loop output addresses".into());
        }
        span = span
            .checked_add(
                stride
                    .checked_mul(count - 1)
                    .ok_or("address span overflow")?,
            )
            .ok_or("address span overflow")?;
    }
    Ok(())
}


/// Interchange a perfect nest of independent serial output loops.
pub fn reorder_ranges(linear: &UOp, order: &[usize]) -> Result<UOp> {
    let linear = complete(linear)?;
    if order.len() < 2 || order.iter().collect::<HashSet<_>>().len() != order.len() {
        return Err("reorder needs distinct axes".into());
    }
    let mut spans = order
        .iter()
        .map(|&axis| range_span(&linear, axis))
        .collect::<Result<Vec<_>>>()?;
    spans.sort_unstable();
    let ranges = spans
        .iter()
        .map(|&(start, _)| linear.src[start].clone())
        .collect::<Vec<_>>();
    for r in &ranges {
        if !matches!(
            r.arg,
            Arg::Range {
                axis: Axis::Loop,
                ..
            }
        ) {
            return Err("only independent output loops can be reordered".into());
        }
        if r.src
            .iter()
            .any(|s| s.toposort().iter().any(|u| ranges.contains(u)))
        {
            return Err("dependent loop bounds cannot be reordered".into());
        }
    }
    for pair in spans.windows(2) {
        if pair[0].1 <= pair[1].1 {
            return Err("axes do not form a perfect loop nest".into());
        }
        if linear.src[pair[0].0 + 1..pair[1].0].iter().any(|u| {
            matches!(
                u.op,
                Ops::Range | Ops::End | Ops::Store | Ops::Load | Ops::Alloc | Ops::Call
            ) || u.toposort().iter().any(|s| ranges.contains(s))
        }) {
            return Err("cannot move operations between loop headers".into());
        }
        if pair[0].1 != pair[1].1 + 1 {
            return Err("operations between loop ends prevent interchange".into());
        }
    }
    let body = &linear.src[spans.last().unwrap().0 + 1..spans.last().unwrap().1];
    let mut writes = HashSet::new();
    let mut stores = vec![];
    for u in body {
        if matches!(u.op, Ops::Call) {
            return Err("unknown effects prevent interchange".into());
        }
        if u.op == Ops::Store {
            let pointer = &u.src[0];
            let base = memory_base(pointer);
            if base.op == Ops::Alloc {
                if !body.iter().any(|node| node == base) {
                    return Err("loop-carried accumulator prevents interchange".into());
                }
            } else {
                if pointer.op != Ops::Index {
                    return Err("external store needs an indexed address".into());
                }
                injective(&pointer.src[1], &ranges)?;
                writes.insert(base.id);
                stores.push(base.id);
            }
        }
    }
    if stores.is_empty() {
        return Err("no independent output stores to interchange".into());
    }
    if stores.iter().collect::<HashSet<_>>().len() != stores.len() {
        return Err("multiple writes to one buffer need dependence analysis".into());
    }
    for u in body {
        if u.op == Ops::Load && writes.contains(&memory_base(&u.src[0]).id) {
            return Err("read-after-write dependence prevents interchange".into());
        }
    }
    let desired = order
        .iter()
        .map(|&id| {
            ranges
                .iter()
                .find(|r| matches!(r.arg, Arg::Range { id: n, .. } if n == id))
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let mut map = HashMap::new();
    let mut result = vec![];
    let mut closed = None;
    for (i, u) in linear.src.iter().enumerate() {
        if let Some(depth) = spans.iter().position(|&(start, _)| start == i) {
            result.push(desired[depth].clone());
            continue;
        }
        if let Some(depth) = spans.iter().position(|&(_, end)| end == i) {
            let value = closed
                .clone()
                .unwrap_or_else(|| rebuild(&u.src[0], &mut map));
            let end = UOp::new(Ops::End, vec![value, desired[depth].clone()], Arg::None);
            map.insert(u.id, end.clone());
            result.push(end.clone());
            closed = Some(end);
            continue;
        }
        result.push(rebuild(u, &mut map));
    }
    let result = UOp::linear(result);
    codegen::verify_lowered(&result)?;
    Ok(result)
}
```

`tests/loop_interchange.rs`を保存する。

```rust
// file="tests/loop_interchange.rs" mode="write"
use chibigrad::optimization::reorder_ranges;
use chibigrad::{Arg, Axis, DType, Ops, UOp, codegen, cpu::CpuProgram};
fn param(index: usize, shape: &[usize]) -> UOp {
    UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index,
            shape: shape.to_vec(),
        },
    )
}
fn end(body: UOp, range: UOp) -> UOp {
    UOp::new(Ops::End, vec![body, range], Arg::None)
}

fn execute(linear: &UOp, inputs: &[Vec<f32>], count: usize) -> Vec<f32> {
    codegen::verify_lowered(linear).unwrap();
    let source = codegen::render(linear, "entry").unwrap();
    let p = CpuProgram::compile(&source, "entry").unwrap();
    let mut output = vec![-999f32; count + 1];
    output[count] = 12345.;
    let mut args = vec![];
    for input in inputs {
        args.extend((input.as_ptr() as u64).to_ne_bytes());
    }
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { p.run(&mut args) }.unwrap();
    assert_eq!(output.pop().unwrap(), 12345., "tail canary");
    output
}

fn matrix_kernel(m: i64, n: i64, k: i64) -> UOp {
    let a = param(0, &[m as usize, k as usize]);
    let b = param(1, &[k as usize, n as usize]);
    let c = param(2, &[m as usize, n as usize]);
    let i = UOp::range(UOp::int(m), 0, Axis::Loop);
    let j = UOp::range(UOp::int(n), 1, Axis::Loop);
    let r = UOp::range(UOp::int(k), 2, Axis::Reduce);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let ptr = alloc.after(vec![init.clone(), r.clone()]);
    let av = a.index(i.mul(&UOp::int(k)).add(&r), UOp::bool(true)).load();
    let bv = b.index(r.mul(&UOp::int(n)).add(&j), UOp::bool(true)).load();
    let update = ptr.store(ptr.load().add(&av.mul(&bv)));
    let ek = end(update.clone(), r.clone());
    let store = c
        .index(i.mul(&UOp::int(n)).add(&j), UOp::bool(true))
        .store(alloc.after(vec![ek.clone()]).load());
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    UOp::linear(vec![i, j, alloc, init, r, update, ek, store, ej, ei])
}

#[test]
fn interchange_executes_an_independent_matrix_kernel() {
    let linear = matrix_kernel(3, 5, 7);
    let a = (0..21).map(|i| (i % 7) as f32 - 2.).collect::<Vec<_>>();
    let b = (0..35).map(|i| (i % 11) as f32 - 5.).collect::<Vec<_>>();
    let expected = execute(&linear, &[a.clone(), b.clone()], 15);
    let optimized = reorder_ranges(&linear, &[1, 0]).unwrap();
    assert_eq!(execute(&optimized, &[a, b], 15), expected);
    assert!(reorder_ranges(&linear, &[1, 1]).is_err());
    assert!(reorder_ranges(&linear, &[2, 0]).is_err());
}
```

```sh
# check step=17.07
cargo test --test loop_interchange
```

3×5×7 の行列積は軸交換後も一致し、重複軸と還元軸は拒否される。

## 17.08 還元状態を保ってループ展開する

定数回数のループを展開し、残りの反復順序と局所累積変数を保持する。分割、交換、展開を組み合わせてネイティブ実行を検証する。

`src/optimization.rs`を追記する。

```rust
// file="src/optimization.rs" mode="append"


/// Unroll constant loop bounds, retaining a serial remainder in original order.
pub fn unroll_range(linear: &UOp, axis: usize, factor: usize) -> Result<UOp> {
    let linear = complete(linear)?;
    if factor == 0 || factor > 256 {
        return Err("unroll factor must be in 1..=256".into());
    }
    let (start, end) = range_span(&linear, axis)?;
    let range = &linear.src[start];
    if !matches!(
        range.arg,
        Arg::Range {
            axis: Axis::Loop | Axis::Reduce,
            ..
        }
    ) {
        return Err("unroll expects a serial loop or reduction".into());
    }
    let lo = constant(&range.src[0])?;
    let hi = constant(&range.src[1])?;
    let count = hi.checked_sub(lo).ok_or("loop count overflow")?;
    if count < 0 {
        return Err("negative loop count".into());
    }
    if factor == 1 {
        return Ok(linear);
    }
    if linear.src[start + 1..end]
        .iter()
        .any(|u| matches!(u.op, Ops::Call))
    {
        return Err("unknown effects prevent unrolling".into());
    }
    let mut next_axis = linear
        .src
        .iter()
        .filter_map(|u| match u.arg {
            Arg::Range { id, .. } => Some(id),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("axis overflow")?;
    let mut next_alloc = linear
        .src
        .iter()
        .filter_map(|u| match u.arg {
            Arg::Allocation { id, .. } => Some(id),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("allocation overflow")?;
    let mut out = linear.src[..start].to_vec();
    let mut base = out
        .iter()
        .map(|u| (u.id, u.clone()))
        .collect::<HashMap<_, _>>();
    let groups = count / factor as i64;
    let Arg::Range { axis: kind, .. } = range.arg else {
        unreachable!()
    };
    let outer = UOp::range(UOp::int(groups), next_axis, kind);
    next_axis = next_axis.checked_add(1).ok_or("axis overflow")?;
    if groups > 0 {
        out.push(outer.clone());
    }
    let mut last = None;
    fn clone_body(
        u: &UOp,
        map: &mut HashMap<u64, UOp>,
        last: &mut Option<UOp>,
        allocation: &mut usize,
        axis: &mut usize,
    ) -> Result<UOp> {
        if let Some(v) = map.get(&u.id) {
            return Ok(v.clone());
        }
        let mut sources = u
            .src
            .iter()
            .map(|s| clone_body(s, map, last, allocation, axis))
            .collect::<Result<Vec<_>>>()?;
        if u.op == Ops::Load
            && let Some(dependency) = last.as_ref()
        {
            sources[0] = sources[0].after(vec![dependency.clone()]);
        }
        let arg = if let Arg::Allocation { dtype, size, .. } = u.arg {
            let id = *allocation;
            *allocation = allocation.checked_add(1).ok_or("allocation overflow")?;
            Arg::Allocation { id, dtype, size }
        } else if let Arg::Range { axis: kind, .. } = u.arg {
            let id = *axis;
            *axis = axis.checked_add(1).ok_or("axis overflow")?;
            Arg::Range { id, axis: kind }
        } else {
            u.arg.clone()
        };
        let result = UOp::new(u.op, sources, arg);
        map.insert(u.id, result.clone());
        if matches!(u.op, Ops::Store | Ops::End) {
            *last = Some(result.clone());
        }
        Ok(result)
    }
    let mut final_body = UOp::int(0);
    for lane in 0..if groups > 0 { factor } else { 0 } {
        let mut map = base.clone();
        map.insert(
            range.id,
            UOp::int(lo)
                .add(&outer.mul(&UOp::int(factor as i64)))
                .add(&UOp::int(lane as i64)),
        );
        for u in &linear.src[start + 1..end] {
            out.push(clone_body(
                u,
                &mut map,
                &mut last,
                &mut next_alloc,
                &mut next_axis,
            )?);
        }
        final_body = clone_body(
            &linear.src[end].src[0],
            &mut map,
            &mut last,
            &mut next_alloc,
            &mut next_axis,
        )?;
    }
    if groups > 0 {
        let end = UOp::new(Ops::End, vec![final_body, outer], Arg::None);
        out.push(end.clone());
        final_body = end;
        last = Some(final_body.clone());
    }
    for value in lo + groups * factor as i64..hi {
        let mut map = base.clone();
        map.insert(range.id, UOp::int(value));
        for u in &linear.src[start + 1..end] {
            out.push(clone_body(
                u,
                &mut map,
                &mut last,
                &mut next_alloc,
                &mut next_axis,
            )?);
        }
        final_body = clone_body(
            &linear.src[end].src[0],
            &mut map,
            &mut last,
            &mut next_alloc,
            &mut next_axis,
        )?;
    }
    base.insert(linear.src[end].id, final_body);
    for u in &linear.src[end + 1..] {
        out.push(rebuild(u, &mut base));
    }
    let result = UOp::linear(out);
    codegen::verify_lowered(&result)?;
    Ok(result)
}
```

`tests/loop_transforms.rs`を保存する。

```rust
// file="tests/loop_transforms.rs" mode="write"
use chibigrad::optimization::{reorder_ranges, split_range};
use chibigrad::{Arg, Axis, Buffer, DType, Ops, Tensor, UOp, codegen, cpu::CpuProgram};
use std::collections::HashMap;
fn param(index: usize, shape: &[usize]) -> UOp {
    UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index,
            shape: shape.to_vec(),
        },
    )
}
fn end(body: UOp, range: UOp) -> UOp {
    UOp::new(Ops::End, vec![body, range], Arg::None)
}

fn execute(linear: &UOp, inputs: &[Vec<f32>], count: usize) -> Vec<f32> {
    codegen::verify_lowered(linear).unwrap();
    let source = codegen::render(linear, "entry").unwrap();
    let p = CpuProgram::compile(&source, "entry").unwrap();
    let mut output = vec![-999f32; count + 1];
    output[count] = 12345.;
    let mut args = vec![];
    for input in inputs {
        args.extend((input.as_ptr() as u64).to_ne_bytes());
    }
    args.extend((output.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { p.run(&mut args) }.unwrap();
    assert_eq!(output.pop().unwrap(), 12345., "tail canary");
    output
}

fn matrix_kernel(m: i64, n: i64, k: i64) -> UOp {
    let a = param(0, &[m as usize, k as usize]);
    let b = param(1, &[k as usize, n as usize]);
    let c = param(2, &[m as usize, n as usize]);
    let i = UOp::range(UOp::int(m), 0, Axis::Loop);
    let j = UOp::range(UOp::int(n), 1, Axis::Loop);
    let r = UOp::range(UOp::int(k), 2, Axis::Reduce);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let ptr = alloc.after(vec![init.clone(), r.clone()]);
    let av = a.index(i.mul(&UOp::int(k)).add(&r), UOp::bool(true)).load();
    let bv = b.index(r.mul(&UOp::int(n)).add(&j), UOp::bool(true)).load();
    let update = ptr.store(ptr.load().add(&av.mul(&bv)));
    let ek = end(update.clone(), r.clone());
    let store = c
        .index(i.mul(&UOp::int(n)).add(&j), UOp::bool(true))
        .store(alloc.after(vec![ek.clone()]).load());
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    UOp::linear(vec![i, j, alloc, init, r, update, ek, store, ej, ei])
}

use chibigrad::optimization::unroll_range;
#[test]
fn reorder_and_unroll_preserve_native_gemm_accumulators() {
    let linear = matrix_kernel(3, 5, 7);
    let a = (0..21).map(|i| ((i % 7) - 2) as f32).collect::<Vec<_>>();
    let b = (0..35).map(|i| ((i % 11) - 5) as f32).collect::<Vec<_>>();
    let expected = (0..15)
        .map(|p| (0..7).map(|k| a[p / 5 * 7 + k] * b[k * 5 + p % 5]).sum())
        .collect::<Vec<f32>>();
    assert_eq!(execute(&linear, &[a.clone(), b.clone()], 15), expected);
    let reordered = reorder_ranges(&linear, &[1, 0]).unwrap();
    assert_eq!(
        reordered
            .src
            .iter()
            .filter(|u| u.op == Ops::Range)
            .take(2)
            .map(|u| u.arg.clone())
            .collect::<Vec<_>>(),
        vec![
            Arg::Range {
                id: 1,
                axis: Axis::Loop
            },
            Arg::Range {
                id: 0,
                axis: Axis::Loop
            }
        ]
    );
    assert_eq!(execute(&reordered, &[a.clone(), b.clone()], 15), expected);
    for factor in [2, 3, 7, 16] {
        let optimized = unroll_range(&linear, 2, factor).unwrap();
        assert_eq!(
            execute(&optimized, &[a.clone(), b.clone()], 15),
            expected,
            "factor {factor}"
        );
    }
    let optimized = unroll_range(&linear, 1, 3).unwrap();
    assert_eq!(execute(&optimized, &[a, b], 15), expected);
}

#[test]
fn split_and_unroll_real_rowsum_with_output_tail() {
    let x = Tensor::parameter("x", &[7, 65]);
    let y = x.sum_axes(&[1], false);
    let input = Buffer::new(
        vec![7, 65],
        (0..455).map(|i| (i % 13) as f32 - 6.).collect(),
    )
    .unwrap();
    let expected = input
        .data
        .chunks(65)
        .map(|r| r.iter().sum::<f32>())
        .collect::<Vec<_>>();
    for factor in [2, 3, 4, 8] {
        let executable = unsafe {
            chibigrad::runtime::compile_many_with_transform(std::slice::from_ref(&y), |linear| {
                let optimized = unroll_range(linear, 1, factor)?;
                split_range(&optimized, 0, 3)
            })
        }
        .unwrap();
        assert_eq!(executable.kernel_count(), 1);
        assert_eq!(
            executable
                .run(std::slice::from_ref(&input), &HashMap::new())
                .unwrap()[0]
                .data,
            expected
        );
    }
}

#[test]
fn full_unroll_and_zero_count_preserve_loop_carried_state() {
    let a = param(0, &[7]);
    let out = param(1, &[1]);
    let r = UOp::range(UOp::int(7), 0, Axis::Reduce);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 4,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let update = alloc.store(
        alloc
            .load()
            .add(&a.index(r.clone(), UOp::bool(true)).load()),
    );
    let e = end(update.clone(), r.clone());
    let store = out
        .index(UOp::int(0), UOp::bool(true))
        .store(alloc.after(vec![e.clone()]).load());
    let linear = UOp::linear(vec![alloc, init, r, update, e, store]);
    let optimized = unroll_range(&linear, 0, 16).unwrap();
    assert!(!optimized.src.iter().any(|u| u.op == Ops::Range));
    assert_eq!(
        execute(&optimized, &[vec![1., 2., 3., 4., 5., 6., 7.]], 1),
        vec![28.]
    );
    let r = UOp::range(UOp::int(0), 8, Axis::Loop);
    let store = out.index(UOp::int(0), UOp::bool(true)).store(UOp::f32(9.));
    let linear = UOp::linear(vec![r.clone(), store.clone(), end(store, r)]);
    let optimized = unroll_range(&linear, 8, 4).unwrap();
    assert!(optimized.toposort().iter().all(|u| u.op != Ops::Store));
}

#[test]
fn rejects_invalid_axes_bounds_factors_and_dependences() {
    let linear = matrix_kernel(3, 5, 7);
    assert!(split_range(&linear, 900, 4).is_err());
    assert!(split_range(&linear, 2, 4).is_err());
    assert!(unroll_range(&linear, 2, 0).is_err());
    assert!(unroll_range(&linear, 2, 257).is_err());
    assert!(unroll_range(&linear, 90, 4).is_err());
    assert!(reorder_ranges(&linear, &[0]).is_err());
    assert!(reorder_ranges(&linear, &[1, 1]).is_err());
    assert!(reorder_ranges(&linear, &[2, 0]).is_err());
    let r = UOp::range(UOp::int(-3), 0, Axis::Loop);
    let negative = UOp::linear(vec![r.clone(), end(UOp::int(0), r)]);
    assert!(split_range(&negative, 0, 4).is_err());
    assert!(unroll_range(&negative, 0, 4).is_err());
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(i.clone(), 1, Axis::Loop);
    let store = param(0, &[9])
        .index(i.mul(&UOp::int(3)).add(&j), UOp::bool(true))
        .store(UOp::f32(1.));
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    let dependent = UOp::linear(vec![i, j, store, ej, ei]);
    assert!(reorder_ranges(&dependent, &[1, 0]).is_err());
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(UOp::int(3), 1, Axis::Loop);
    let output = param(0, &[9]);
    let address = output.index(i.mul(&UOp::int(3)).add(&j), UOp::bool(true));
    let store = address.store(address.load().add(&UOp::f32(1.)));
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    assert!(reorder_ranges(&UOp::linear(vec![i, j, store, ej, ei]), &[1, 0]).is_err());
}

#[test]
fn compose_split_reorder_unroll_with_a_partial_tile() {
    let input = param(0, &[17]);
    let output = param(1, &[17]);
    let r = UOp::range(UOp::int(17), 0, Axis::Loop);
    let store = output.index(r.clone(), UOp::bool(true)).store(
        input
            .index(r.clone(), UOp::bool(true))
            .load()
            .add(&UOp::f32(2.)),
    );
    let linear = UOp::linear(vec![r.clone(), store.clone(), end(store, r)]);
    let split = split_range(&linear, 0, 4).unwrap();
    let reordered = reorder_ranges(&split, &[2, 1]).unwrap();
    let optimized = unroll_range(&reordered, 2, 3).unwrap();
    let data = (0..17).map(|i| i as f32).collect::<Vec<_>>();
    assert_eq!(
        execute(&optimized, &[data], 17),
        (0..17).map(|i| i as f32 + 2.).collect::<Vec<_>>()
    );
}
```

```sh
# check step=17.08
cargo test --test loop_transforms
```

行列積、行和、完全展開、0 回ループ、端数タイルの 5 テストが通る。

## 17.09 独立した出力を CPU ワーカーへ配る

各ワーカーへ専用の出力とロード済みコードを渡す。開始と終了は 2 個の i64 を持つ制御バッファから LOAD するため、変数や記号形状はまだ必要ない。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"

pub mod parallel;
```

`src/parallel.rs`を保存する。

```rust
// file="src/parallel.rs" mode="write"
use crate::{Arg, Axis, Buffer, Ops, Result, UOp, codegen, cpu};
use std::{collections::HashMap, sync::Arc};

/// A compiled kernel whose workers own separate output allocations.
pub struct ParallelProgram {
    object: Arc<Vec<u8>>,
    entry: String,
    buffers: Vec<(usize, Vec<usize>)>,
    output_slot: usize,
    bounds_slot: usize,
    extent: i64,
    workers: usize,
}

#[derive(Debug)]
pub struct WorkerRun {
    pub output: Buffer,
    pub thread_ids: Vec<std::thread::ThreadId>,
    pub partitions: Vec<std::ops::Range<usize>>,
}

impl ParallelProgram {
    pub fn compile(linear: &UOp, axis: usize, workers: usize) -> Result<Self> {
        let linear = crate::optimization::complete(linear)?;
        if workers == 0 || workers > 256 {
            return Err("workers must be in 1..=256".into());
        }
        let range = linear
            .src
            .iter()
            .find(|u| u.op == Ops::Range && matches!(u.arg,Arg::Range { id, .. } if id == axis))
            .ok_or("unknown worker axis")?;
        if !matches!(
            range.arg,
            Arg::Range {
                axis: Axis::Loop,
                ..
            }
        ) || range.src[0] != UOp::int(0)
        {
            return Err("worker axis must be a zero-based output loop".into());
        }
        let Arg::Int(extent) = range.src[1].arg else {
            return Err("worker partition requires a constant output count".into());
        };
        if extent < 0 {
            return Err("negative output count".into());
        }
        let begin = linear.src.iter().position(|u| u == range).unwrap();
        let finish = linear
            .src
            .iter()
            .position(|u| u.op == Ops::End && u.src.last() == Some(range))
            .ok_or("missing worker range END")?;
        if linear.src[..begin]
            .iter()
            .any(|u| matches!(u.op, Ops::Range | Ops::Store | Ops::Call))
            || linear.src[finish + 1..]
                .iter()
                .any(|u| matches!(u.op, Ops::Range | Ops::Store | Ops::Call))
        {
            return Err("worker axis must enclose every kernel effect".into());
        }
        let body = &linear.src[begin + 1..finish];
        let all = linear.toposort();
        if all.iter().any(|u| matches!(u.op, Ops::Call)) {
            return Err("unknown effects prevent parallel execution".into());
        }
        let mut stores = vec![];
        for u in &all {
            if u.op != Ops::Store {
                continue;
            }
            let mut pointer = &u.src[0];
            while pointer.op == Ops::After {
                pointer = &pointer.src[0];
            }
            if pointer.op == Ops::Alloc {
                if !body.iter().any(|node| node == pointer) {
                    return Err("loop-carried accumulator prevents parallel execution".into());
                }
                continue;
            }
            if !matches!(pointer.arg, Arg::None | Arg::DType(crate::DType::F32)) {
                return Err("worker output requires scalar F32 stores".into());
            }
            if pointer.op != Ops::Index || pointer.src[1] != *range {
                return Err("worker output must be indexed directly by the partition axis".into());
            }
            stores.push(pointer.src[0].clone());
        }
        if stores.len() != 1 {
            return Err("worker kernel must have one contiguous output store".into());
        }
        let output = &stores[0];
        for u in &all {
            if u.op == Ops::Load {
                let mut pointer = &u.src[0];
                while matches!(pointer.op, Ops::After | Ops::Index) {
                    pointer = &pointer.src[0];
                }
                if pointer == output {
                    return Err("workers cannot read their output buffer".into());
                }
            }
        }
        let sig = codegen::signature(&linear)?;
        let output_slot = sig
            .buffers
            .iter()
            .find(|(_, u)| u == output)
            .ok_or("output missing from signature")?
            .0;
        let buffers = sig
            .buffers
            .iter()
            .map(|(slot, u)| Ok((*slot, buffer_shape(u)?)))
            .collect::<Result<Vec<_>>>()?;
        if crate::uop::numel(&buffer_shape(output)?) != extent as usize {
            return Err("output shape must match worker axis extent".into());
        }
        let bounds_slot = sig
            .buffers
            .iter()
            .map(|(slot, _)| *slot)
            .max()
            .map_or(0, |slot| slot + 1);
        let bounds = UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index: bounds_slot,
                shape: vec![2],
            },
        );
        let bound = |index| {
            let pointer = UOp::new(
                Ops::Index,
                vec![bounds.clone(), UOp::int(index), UOp::bool(true)],
                Arg::DType(crate::DType::I64),
            );
            UOp::new(Ops::Load, vec![pointer], Arg::DType(crate::DType::I64))
        };
        let start = bound(0);
        let stop = bound(1);
        let replacement = UOp::new(
            Ops::Range,
            vec![start.clone(), stop.clone()],
            range.arg.clone(),
        );
        fn rewrite(u: &UOp, map: &mut HashMap<u64, UOp>) -> UOp {
            if let Some(v) = map.get(&u.id) {
                return v.clone();
            }
            let sources = u.src.iter().map(|s| rewrite(s, map)).collect();
            let v = UOp::new(u.op, sources, u.arg.clone());
            map.insert(u.id, v.clone());
            v
        }
        let mut map = HashMap::from([(range.id, replacement)]);
        let partitioned = UOp::linear(linear.src.iter().map(|u| rewrite(u, &mut map)).collect());
        codegen::verify_lowered(&partitioned)?;
        let partition_sig = codegen::signature(&partitioned)?;
        for (slot, original) in &sig.buffers {
            let replacement = partition_sig
                .buffers
                .iter()
                .find(|(_, u)| u == original)
                .ok_or("buffer binding changed")?;
            if replacement.0 != *slot {
                return Err("worker partition changed buffer ABI".into());
            }
        }
        let entry = "chibi_parallel_entry".to_owned();
        let source = codegen::render(&partitioned, &entry)?;
        let object = Arc::new(cpu::compile_object(&source)?);
        Ok(Self {
            object,
            entry,
            buffers,
            output_slot,
            bounds_slot,
            extent,
            workers,
        })
    }

    pub fn object(&self) -> &[u8] {
        &self.object
    }
    pub fn worker_count(&self) -> usize {
        self.workers.min(self.extent as usize)
    }

    /// Buffers bind the original kernel's signature slots.
    pub fn run(&self, inputs: &[(usize, Buffer)]) -> Result<WorkerRun> {
        let input_count = inputs.len();
        let inputs = inputs
            .iter()
            .map(|(slot, buffer)| (*slot, buffer.clone()))
            .collect::<HashMap<_, _>>();
        if inputs.len() != input_count
            || inputs.len() != self.buffers.len() - 1
            || inputs.contains_key(&self.output_slot)
        {
            return Err("worker inputs must bind every read buffer exactly once".into());
        }
        for (slot, shape) in &self.buffers {
            if *slot == self.output_slot {
                continue;
            }
            let b = inputs.get(slot).ok_or("missing worker buffer")?;
            if b.shape != *shape || b.data.len() != crate::uop::numel(shape) {
                return Err(format!("worker buffer {slot} shape mismatch"));
            }
        }
        let count = self.worker_count();
        let output_shape = self
            .buffers
            .iter()
            .find(|(slot, _)| *slot == self.output_slot)
            .unwrap()
            .1
            .clone();
        let mut output = Buffer::new(output_shape, vec![0.; self.extent as usize])?;
        if count == 0 {
            return Ok(WorkerRun {
                output,
                thread_ids: vec![],
                partitions: vec![],
            });
        }
        let inputs = Arc::new(inputs);
        let mut handles = vec![];
        for worker in 0..count {
            let lo = self.extent as usize * worker / count;
            let hi = self.extent as usize * (worker + 1) / count;
            let object = self.object.clone();
            let entry = self.entry.clone();
            let input = inputs.clone();
            let output_slot = self.output_slot;
            let bounds_slot = self.bounds_slot;
            let extent = self.extent as usize;
            let arg_count = self
                .buffers
                .iter()
                .map(|b| b.0)
                .chain([bounds_slot])
                .max()
                .unwrap()
                + 1;
            handles.push(std::thread::spawn(move || -> Result<_> {
                let native = cpu::CpuProgram::load(&object, &entry)?;
                let mut output = vec![0f32; extent];
                let mut packed = vec![0u8; arg_count * 8];
                for (&slot, b) in input.iter() {
                    packed[slot * 8..slot * 8 + 8]
                        .copy_from_slice(&(b.data.as_ptr() as u64).to_ne_bytes());
                }
                packed[output_slot * 8..output_slot * 8 + 8]
                    .copy_from_slice(&(output.as_mut_ptr() as u64).to_ne_bytes());
                let bounds = [lo as i64, hi as i64];
                packed[bounds_slot * 8..bounds_slot * 8 + 8]
                    .copy_from_slice(&(bounds.as_ptr() as u64).to_ne_bytes());
                unsafe { native.run(&mut packed) }?;
                Ok((lo..hi, output, std::thread::current().id()))
            }));
        }
        let mut partitions = vec![];
        let mut thread_ids = vec![];
        let mut error = None;
        for handle in handles {
            match handle
                .join()
                .map_err(|_| "CPU worker panicked".to_owned())
                .and_then(|result| result)
            {
                Ok((range, data, id)) => {
                    output.data[range.clone()].copy_from_slice(&data[range.clone()]);
                    partitions.push(range);
                    thread_ids.push(id);
                }
                Err(message) => {
                    error.get_or_insert(message);
                }
            }
        }
        if let Some(message) = error {
            return Err(message);
        }
        Ok(WorkerRun {
            output,
            thread_ids,
            partitions,
        })
    }
}

fn buffer_shape(u: &UOp) -> Result<Vec<usize>> {
    match &u.arg {
        Arg::Slot { shape, .. } => Ok(shape.clone()),
        _ => u.shape(),
    }
}
```

`tests/cpu_workers.rs`を保存する。

```rust
// file="tests/cpu_workers.rs" mode="write"
use chibigrad::optimization::{reorder_ranges, split_range, unroll_range};
use chibigrad::parallel::ParallelProgram;
use chibigrad::{Arg, Axis, Buffer, DType, Ops, Tensor, UOp, codegen};
use std::collections::HashSet;
fn param(index: usize, shape: &[usize]) -> UOp {
    UOp::new(
        Ops::Param,
        vec![],
        Arg::Slot {
            index,
            shape: shape.to_vec(),
        },
    )
}
fn end(body: UOp, range: UOp) -> UOp {
    UOp::new(Ops::End, vec![body, range], Arg::None)
}

#[test]
fn workers_execute_native_rowsum_and_rebind_without_recompilation() {
    let x = Tensor::parameter("x", &[7, 65]);
    let y = x.sum_axes(&[1], false);
    let executable = y.compile().unwrap();
    let linear = &executable.programs()[0].src[1];
    let input_slot = codegen::signature(linear)
        .unwrap()
        .buffers
        .into_iter()
        .find(|(_, u)| matches!(&u.arg,Arg::Buffer{name,..} if name=="x"))
        .unwrap()
        .0;
    for workers in [1, 2, 3, 16] {
        let program = ParallelProgram::compile(linear, 0, workers).unwrap();
        let object = program.object().to_vec();
        for offset in [0., 3., -2.] {
            let buffer = Buffer::new(
                vec![7, 65],
                (0..455).map(|i| (i % 11) as f32 + offset).collect(),
            )
            .unwrap();
            let expected = buffer
                .data
                .chunks(65)
                .map(|r| r.iter().sum::<f32>())
                .collect::<Vec<_>>();
            let result = program.run(&[(input_slot, buffer)]).unwrap();
            assert_eq!(result.output.data, expected);
            assert_eq!(
                result.thread_ids.iter().collect::<HashSet<_>>().len(),
                workers.min(7)
            );
            assert!(
                result
                    .thread_ids
                    .iter()
                    .all(|&id| id != std::thread::current().id())
            );
            assert_eq!(
                result
                    .partitions
                    .iter()
                    .flat_map(|r| r.clone())
                    .collect::<Vec<_>>(),
                (0..7).collect::<Vec<_>>()
            );
            assert_eq!(program.object(), object);
        }
    }
}

#[test]
fn workers_execute_flattened_gemm_with_private_reduction_state() {
    let a = Tensor::parameter("a", &[3, 7]);
    let b = Tensor::parameter("b", &[7, 5]);
    let executable = a.matmul(&b).compile().unwrap();
    let linear = &executable.programs()[0].src[1];
    let av = (0..21).map(|i| (i % 7) as f32 - 2.).collect::<Vec<_>>();
    let bv = (0..35).map(|i| (i % 11) as f32 - 5.).collect::<Vec<_>>();
    let mut buffers = vec![];
    for (slot, u) in codegen::signature(linear).unwrap().buffers {
        if let Arg::Buffer { name, shape, .. } = &u.arg {
            if name == "a" {
                buffers.push((slot, Buffer::new(shape.clone(), av.clone()).unwrap()));
            }
            if name == "b" {
                buffers.push((slot, Buffer::new(shape.clone(), bv.clone()).unwrap()));
            }
        }
    }
    let expected = (0..15)
        .map(|p| {
            (0..7)
                .map(|k| av[p / 5 * 7 + k] * bv[k * 5 + p % 5])
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    for workers in [2, 4, 32] {
        let program = ParallelProgram::compile(linear, 0, workers).unwrap();
        assert_eq!(program.run(&buffers).unwrap().output.data, expected);
    }
}

#[test]
fn splitting_and_workers_reject_a_cross_iteration_accumulator() {
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(UOp::int(3), 1, Axis::Loop);
    let alloc = UOp::new(
        Ops::Alloc,
        vec![],
        Arg::Allocation {
            id: 0,
            dtype: DType::F32,
            size: 1,
        },
    );
    let init = alloc.store(UOp::f32(0.));
    let update = alloc.store(alloc.load().add(&UOp::f32(1.)));
    let ej = end(update.clone(), j.clone());
    let output = param(0, &[3])
        .index(i.clone(), UOp::bool(true))
        .store(alloc.after(vec![ej.clone()]).load());
    let ei = end(output.clone(), i.clone());
    let linear = UOp::linear(vec![alloc, init, i, j, update, ej, output, ei]);
    assert!(split_range(&linear, 0, 2).is_err());
    assert!(reorder_ranges(&linear, &[1, 0]).is_err());
    assert!(ParallelProgram::compile(&linear, 0, 2).is_err());
}

#[test]
fn unknown_effects_and_overlapping_output_addresses_are_rejected() {
    let i = UOp::range(UOp::int(3), 0, Axis::Loop);
    let j = UOp::range(UOp::int(3), 1, Axis::Loop);
    let output = param(0, &[9]);
    let store = output.index(i.add(&j), UOp::bool(true)).store(UOp::f32(1.));
    let ej = end(store.clone(), j.clone());
    let ei = end(ej.clone(), i.clone());
    let overlap = UOp::linear(vec![i.clone(), j, store, ej, ei]);
    assert!(reorder_ranges(&overlap, &[1, 0]).is_err());
    let call = UOp::new(Ops::Call, vec![], Arg::Text("opaque_effect".to_owned()));
    let effect = UOp::linear(vec![i.clone(), call.clone(), end(call, i)]);
    assert!(split_range(&effect, 0, 2).is_err());
    assert!(unroll_range(&effect, 0, 2).is_err());
    assert!(ParallelProgram::compile(&effect, 0, 2).is_err());
}

#[test]
fn worker_count_is_bounded_by_outputs_and_empty_outputs_do_not_spawn() {
    for n in [0, 1] {
        let output = param(0, &[n]);
        let r = UOp::range(UOp::int(n as i64), 0, Axis::Loop);
        let store = output.index(r.clone(), UOp::bool(true)).store(UOp::f32(7.));
        let linear = UOp::linear(vec![r.clone(), store.clone(), end(store, r)]);
        let program = ParallelProgram::compile(&linear, 0, 16).unwrap();
        assert_eq!(program.worker_count(), n);
        let result = program.run(&[]).unwrap();
        assert_eq!(result.output.data, vec![7.; n]);
        assert_eq!(result.thread_ids.len(), n);
    }
}
```

```sh
# check step=17.09
cargo test --test cpu_workers
```

行和と行列積が再束縛後も一致し、範囲は重複なく全出力を覆う。空出力はスレッドを作らない。

## 17.10 同じ LOAD と STORE を4要素のvectorへ広げる

F32x4 を加え、C の vector 型とGPUのfloat4へ描画する。INDEXの要素型とLOAD・STOREの型を揃え、同じ構造で4要素ずつ処理する。GPU用には最小のBuilderとグローバル添字を追加する。

`src/codegen.rs` の `fn ctype`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn ctype"
fn ctype(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::F32x4 => "float4",
        DType::I32 => "int",
        DType::I64 => "long long",
        DType::Bool => "int",
        DType::Ptr => "void*",
        DType::Void => "void",
    }
}
```

`src/codegen.rs` の `fn render_impl`を置き換える。

```rust
// file="src/codegen.rs" mode="replace" item="fn render_impl"
fn render_impl(linear: &UOp, entry: &str, status_slot: Option<usize>) -> Result<String> {
    let mut all = vec![];
    let mut seen = HashSet::new();
    let mut effects = HashSet::new();
    let mut scopes = vec![];
    fn add(
        u: &UOp,
        out: &mut Vec<UOp>,
        seen: &mut HashSet<u64>,
        effects: &mut HashSet<u64>,
        scopes: &mut Vec<HashSet<u64>>,
    ) {
        if seen.contains(&u.id) || effects.contains(&u.id) {
            return;
        }
        for src in u.ordered_sources() {
            add(src, out, seen, effects, scopes)
        }
        if u.op == Ops::Range {
            scopes.push(seen.clone());
        }
        seen.insert(u.id);
        out.push(u.clone());
        if matches!(u.op, Ops::Range | Ops::End | Ops::Store | Ops::Call) {
            effects.insert(u.id);
        }
        if u.op == Ops::End {
            if let Some(parent) = scopes.pop() {
                *seen = parent;
            }
        }
    }
    for u in &linear.src {
        add(u, &mut all, &mut seen, &mut effects, &mut scopes)
    }
    let complete = UOp::linear(all);
    let linear = &complete;
    verify_lowered(linear)?;

    if !identifier(entry) {
        return Err("invalid C entry identifier".into());
    }
    let sig = signature(linear)?;
    let mut text = String::from(
        "extern float exp2f(float);\nextern float log2f(float);\nextern float sinf(float);\nextern float sqrtf(float);\nextern float fmaxf(float,float);\nstatic float chibi_float(unsigned int x) { union { unsigned int u; float f; } b; b.u=x; return b.f; }\nstatic long long chibi_div(long long a,long long b) { long long q=a/b, r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }\nstatic long long chibi_mod(long long a,long long b) { long long r=a%b; return r+(((r!=0)&&((r<0)!=(b<0)))?b:0); }\n",
    );
    text.push_str(
        "static int chibi_int(unsigned int x) { union { unsigned int u; int i; } b; b.u=x; return b.i; }\nstatic unsigned int chibi_bits(float x) { union { unsigned int u; float f; } b; b.f=x; return b.u; }\nstatic int chibi_shr(int a,int b) { unsigned int u=(unsigned int)a; return chibi_int((u>>b)|((a<0&&b)?(~0u<<(32-b)):0u)); }\nstatic long long chibi_shr64(long long a,long long b) { unsigned long long u=(unsigned long long)a; return (long long)((u>>b)|((a<0&&b)?(~0ull<<(64-b)):0ull)); }\nstatic float chibi_max(float a,float b) { if(a!=a||b!=b)return chibi_float(2143289344u); if(a==0.0f&&b==0.0f)return 0.0f; return a>b?a:b; }\n",
    );
    let mut functions: HashMap<String, (DType, Vec<DType>)> = HashMap::new();
    for u in &linear.src {
        let (body, args) = if u.op == Ops::Call {
            let b = u.src.first().ok_or("CALL lacks body")?;
            (b, &u.src[1..])
        } else if u.op == Ops::CustomFunction && !u.src.is_empty() {
            (u, &u.src[..])
        } else {
            continue;
        };
        let Arg::Function { name, dtype } = &body.arg else {
            continue;
        };
        if !identifier(name) {
            return Err("invalid C function identifier".into());
        }
        let types = args.iter().map(UOp::dtype).collect::<Vec<_>>();
        if functions
            .get(name)
            .is_some_and(|old| old != &(*dtype, types.clone()))
        {
            return Err(format!("inconsistent callback signature for {name}"));
        }
        functions.insert(name.clone(), (*dtype, types));
    }
    let mut function_names = functions.keys().collect::<Vec<_>>();
    function_names.sort();
    for name in function_names {
        let (ret, args) = &functions[name];
        text.push_str(&format!(
            "extern {} {}({});\n",
            ctype(*ret),
            name,
            if args.is_empty() {
                "void".into()
            } else {
                args.iter().map(|d| ctype(*d)).collect::<Vec<_>>().join(",")
            }
        ));
    }
    text.push_str("typedef float float4 __attribute__((ext_vector_type(4),aligned(1)));\n");
    text.push_str(&format!("void {entry}(unsigned char *args) {{\n"));
    let mut values = HashMap::new();
    let mut addresses: HashMap<u64, (String, String, String, DType)> = HashMap::new();
    for (slot, u) in &sig.buffers {
        let name = format!("b{}", u.id);
        text.push_str(&format!(
            "  void *{name}=(void*)(unsigned long long)((unsigned long long*)args)[{slot}];\n"
        ));
        values.insert(u.id, name);
    }
    let mut depth = 1;
    let mut occurrence = 0usize;
    let mut value_scopes = vec![];
    let mut address_scopes = vec![];
    for u in &linear.src {
        let get = |s: &UOp| -> Result<String> {
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| format!("{:?} {} used before definition", s.op, s.id))
        };
        let at = |i: usize| -> Result<String> { get(u.src.get(i).ok_or("missing source")?) };
        let indent = "  ".repeat(depth);
        if let Some(slot) = status_slot {
            let reject = format!(
                "{{ ((int*)(unsigned long long)((unsigned long long*)args)[{slot}])[0]=1; return; }}"
            );
            let a = || at(0);
            let b = || at(1);
            let guard = match u.op {
                Ops::Shl | Ops::Shr => Some(format!(
                    "({}<0 || {}>={})",
                    b()?,
                    b()?,
                    if u.dtype() == DType::I32 { 32 } else { 64 }
                )),
                Ops::Div | Ops::Mod if matches!(u.dtype(), DType::I32 | DType::I64) => {
                    Some(format!(
                        "({}==0 || ({}=={} && {}==-1))",
                        b()?,
                        a()?,
                        if u.dtype() == DType::I32 {
                            "(-2147483647-1)"
                        } else {
                            "(-9223372036854775807ll-1)"
                        },
                        b()?
                    ))
                }
                Ops::Cast
                    if u.src[0].dtype() == DType::F32
                        && matches!(u.dtype(), DType::I32 | DType::I64) =>
                {
                    let limit = if u.dtype() == DType::I32 {
                        "2147483648.0f"
                    } else {
                        "9223372036854775808.0f"
                    };
                    Some(format!("!({} >= -{limit} && {} < {limit})", a()?, a()?))
                }
                Ops::Add | Ops::Sub | Ops::Mul if u.dtype() == DType::I64 => {
                    let operation = match u.op {
                        Ops::Add => "add",
                        Ops::Sub => "sub",
                        _ => "mul",
                    };
                    text.push_str(&format!("{indent}long long checked{};\n", u.id));
                    Some(format!(
                        "__builtin_{operation}_overflow({}, {}, &checked{})",
                        a()?,
                        b()?,
                        u.id
                    ))
                }
                Ops::Neg if u.dtype() == DType::I64 => {
                    Some(format!("({}==(-9223372036854775807ll-1))", a()?))
                }
                _ => None,
            };
            if let Some(condition) = guard {
                text.push_str(&format!("{indent}if ({condition}) {reject}\n"));
            }
        }
        let expression = match (&u.op, &u.arg) {
            (Ops::Const, Arg::Float(bits)) => {
                let bits = (bits.value() as f32).to_bits();
                values.insert(u.id, format!("chibi_float({bits}u)"));
                continue;
            }
            (Ops::Const, Arg::I32(v)) => {
                values.insert(u.id, format!("((int){}u)", *v as u32));
                continue;
            }
            (Ops::Const, Arg::Int(v)) => {
                values.insert(u.id, format!("((long long){}ull)", *v as u64));
                continue;
            }
            (Ops::Const, Arg::Bool(v)) => {
                values.insert(u.id, if *v { "1" } else { "0" }.into());
                continue;
            }
            (Ops::Buffer, _) | (Ops::Param, _) => continue,
            (Ops::Alloc, Arg::Allocation { dtype, size, .. }) => {
                let name = format!("a{}", u.id);
                text.push_str(&format!(
                    "{indent}{} {name}[{}];\n",
                    ctype(*dtype),
                    size.max(&1)
                ));
                values.insert(u.id, name);
                continue;
            }
            (Ops::After, _) => {
                values.insert(u.id, at(0)?);
                if let Some(addr) = addresses.get(&u.src[0].id).cloned() {
                    addresses.insert(u.id, addr);
                }
                continue;
            }
            (Ops::Index, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                addresses.insert(u.id, (at(0)?, at(1)?, at(2)?, dtype));
                values.insert(
                    u.id,
                    format!("(({}*){})+({})", ctype(dtype), at(0)?, at(1)?),
                );
                continue;
            }
            (Ops::Load, _) => {
                let dtype = if let Arg::DType(dt) = u.arg {
                    dt
                } else {
                    DType::F32
                };
                if let Some((base, index, gate, _)) = addresses.get(&u.src[0].id) {
                    format!(
                        "({gate} ? (({}*){base})[{index}] : ({})0)",
                        ctype(dtype),
                        ctype(dtype)
                    )
                } else {
                    format!("(({}*){})[0]", ctype(dtype), at(0)?)
                }
            }
            (Ops::Store, _) => {
                let value = at(1)?;
                if status_slot.is_some_and(
                    |slot| matches!(u.src[0].arg, Arg::Slot { index,.. } if index == slot),
                ) {
                    text.push_str(&format!(
                        "{indent}if (!({value})) {{ ((int*){})[0]=1; return; }}\n",
                        at(0)?
                    ));
                    values.insert(u.id, "0".into());
                    continue;
                }
                if let Some((base, index, gate, dtype)) = addresses.get(&u.src[0].id) {
                    text.push_str(&format!(
                        "{indent}if ({gate}) (({}*){base})[{index}]={value};\n",
                        ctype(*dtype)
                    ));
                } else {
                    let dtype = if let Arg::DType(dt) = u.arg {
                        dt
                    } else {
                        u.src[1].dtype()
                    };
                    text.push_str(&format!(
                        "{indent}(({}*){})[0]={value};\n",
                        ctype(dtype),
                        at(0)?
                    ));
                }
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Range, _) => {
                value_scopes.push(values.clone());
                address_scopes.push(addresses.clone());
                let name = format!("r{}", u.id);
                text.push_str(&format!(
                    "{indent}for (long long {name}={}; {name}<{}; {name}++) {{\n",
                    at(0)?,
                    at(1)?
                ));
                values.insert(u.id, name);
                depth += 1;
                continue;
            }
            (Ops::End, _) => {
                if depth == 1 {
                    return Err("extra END".into());
                }
                depth -= 1;
                text.push_str(&format!("{}}}\n", "  ".repeat(depth)));
                values = value_scopes.pop().ok_or("missing scope")?;
                addresses = address_scopes.pop().ok_or("missing address scope")?;
                values.insert(u.id, "0".into());
                continue;
            }
            (Ops::Cast, Arg::DType(DType::Bool)) => format!("({}!=0)", at(0)?),
            (Ops::Cast, Arg::DType(dtype)) => {
                format!("(({})({}))", ctype(*dtype), at(0)?)
            }
            (Ops::Bitcast, Arg::DType(dt)) if u.src[0].dtype() == *dt => at(0)?,
            (Ops::Bitcast, Arg::DType(DType::I32)) => {
                format!("chibi_int(chibi_bits({}))", at(0)?)
            }
            (Ops::Bitcast, Arg::DType(DType::F32)) => {
                format!("chibi_float((unsigned int)({}))", at(0)?)
            }
            (Ops::Shl, _) => {
                if u.dtype() == DType::I32 {
                    format!("chibi_int((unsigned int)({}) << ({}))", at(0)?, at(1)?)
                } else {
                    format!(
                        "((long long)((unsigned long long)({}) << ({})))",
                        at(0)?,
                        at(1)?
                    )
                }
            }
            (Ops::Shr, _) => {
                format!(
                    "{}({},{})",
                    if u.dtype() == DType::I32 {
                        "chibi_shr"
                    } else {
                        "chibi_shr64"
                    },
                    at(0)?,
                    at(1)?
                )
            }
            (Ops::Neg, _) => {
                if u.dtype() == DType::I32 {
                    format!("((int)(0u-(unsigned int)({})))", at(0)?)
                } else {
                    format!("(-({}))", at(0)?)
                }
            }
            (Ops::Exp2, _) => format!("exp2f({})", at(0)?),
            (Ops::Log2, _) => format!("log2f({})", at(0)?),
            (Ops::Sin, _) => format!("sinf({})", at(0)?),
            (Ops::Sqrt, _) => format!("sqrtf({})", at(0)?),
            (Ops::Where, _) => format!("({}?{}:{})", at(0)?, at(1)?, at(2)?),
            (Ops::Max, _) if u.dtype() == DType::F32 => {
                format!("chibi_max({},{})", at(0)?, at(1)?)
            }
            (Ops::Max, _) => format!("({}>{}?{}:{})", at(0)?, at(1)?, at(0)?, at(1)?),
            (Ops::Div, _) if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", at(0)?, at(1)?)
            }
            (Ops::Mod, _) => format!("chibi_mod({},{})", at(0)?, at(1)?),
            (op, _) if is_binary(*op) => {
                let symbol = match op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&",
                    Ops::Or => "|",
                    Ops::Xor => "^",
                    _ => return Err("unsupported ALU".into()),
                };
                if u.dtype() == DType::I32 && matches!(op, Ops::Add | Ops::Sub | Ops::Mul) {
                    format!(
                        "((int)((unsigned int)({}) {symbol} (unsigned int)({})))",
                        at(0)?,
                        at(1)?
                    )
                } else {
                    format!("({} {symbol} {})", at(0)?, at(1)?)
                }
            }
            (Ops::Call, _) => {
                let body = u.src.first().ok_or("CALL body missing")?;
                let Arg::Function { name, .. } = &body.arg else {
                    return Err("scalar CALL must target CUSTOM_FUNCTION".into());
                };
                format!(
                    "{}({})",
                    name,
                    u.src[1..]
                        .iter()
                        .map(get)
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                )
            }
            (Ops::CustomFunction, Arg::Function { name, .. }) => {
                if u.src.is_empty() {
                    values.insert(u.id, name.clone());
                    continue;
                }
                format!(
                    "{}({})",
                    name,
                    u.src.iter().map(get).collect::<Result<Vec<_>>>()?.join(",")
                )
            }
            (Ops::Sink, _) => continue,
            _ => return Err(format!("C renderer does not support {:?}", u.op)),
        };
        if u.dtype() == DType::Void {
            text.push_str(&format!("{indent}{expression};\n"));
            values.insert(u.id, "0".into());
        } else {
            let name = format!("v{}_{}", u.id, occurrence);
            occurrence += 1;
            text.push_str(&format!(
                "{indent}{} {name}={expression};\n",
                ctype(u.dtype())
            ));
            values.insert(u.id, name);
        }
    }
    if depth != 1 {
        return Err("unclosed C loop".into());
    }
    text.push_str("}\n");
    Ok(text)
}
```

`src/gpu.rs`を追記する。

```rust
// file="src/gpu.rs" mode="append"

use crate::{Arg,DType,Ops,UOp};

use std::collections::HashSet;


pub(crate) struct Builder {
    ops: Vec<UOp>,
    seen: HashSet<u64>,
}

impl Builder {
    pub(crate) fn new() -> Self {
        Self {
            ops: vec![],
            seen: HashSet::new(),
        }
    }
    pub(crate) fn emit(&mut self, u: UOp) -> UOp {
        for s in &u.src {
            if !self.seen.contains(&s.id) {
                self.emit(s.clone());
            }
        }
        if self.seen.insert(u.id) {
            self.ops.push(u.clone());
        }
        u
    }
    pub(crate) fn special(&mut self, name: &str) -> UOp {
        self.emit(UOp::new(Ops::Special, vec![], Arg::Text(name.into())))
    }
    pub(crate) fn param(&mut self, index: usize, size: usize) -> UOp {
        self.emit(UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index,
                shape: vec![size],
            },
        ))
    }
    pub(crate) fn finish(self) -> UOp {
        UOp::linear(self.ops)
    }
}

/// Strided lane accumulation and shared-memory tree reduction, expressed only with common UOps.

pub fn vector_add(count: usize, local: usize) -> Result<(UOp, Launch), String> {
    if count == 0 || !count.is_multiple_of(4) || local == 0 {
        return Err("vector add requires a nonzero multiple of four elements".into());
    }
    let mut b = Builder::new();
    let a = b.param(0, count);
    let rhs = b.param(1, count);
    let out = b.param(2, count);
    let gid = b.special("gid0");
    let valid = gid.lt(&UOp::int((count / 4) as i64));
    let index = |p: UOp| {
        UOp::new(
            Ops::Index,
            vec![p, gid.clone(), valid.clone()],
            Arg::DType(DType::F32x4),
        )
    };
    let a = b.emit(UOp::new(
        Ops::Load,
        vec![index(a)],
        Arg::DType(DType::F32x4),
    ));
    let rhs = b.emit(UOp::new(
        Ops::Load,
        vec![index(rhs)],
        Arg::DType(DType::F32x4),
    ));
    b.emit(index(out).store(a.add(&rhs)));
    Ok((
        b.finish(),
        Launch {
            groups: [(count / 4).div_ceil(local), 1, 1],
            local: [local, 1, 1],
        },
    ))
}
```

`src/renderers.rs` の `fn ident`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="fn ident"
fn ident(name: &str) -> Result<(), String> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.as_bytes()[0].is_ascii_digit()
    {
        Err("invalid GPU identifier".into())
    } else {
        Ok(())
    }
}
```

`src/renderers.rs` の `fn ty`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="fn ty"
fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F32x4 => "float4",
        DType::I64 => "long",
        DType::Bool => "bool",
        DType::Ptr => "float*",
        DType::Void => "void",
        _ => "int",
    }
}
```

`src/renderers.rs` の `pub fn render`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="pub fn render"
pub fn render(
    linear: &UOp,
    entry: &str,
    dialect: Dialect,
    launch: Launch,
) -> Result<Kernel, String> {
    ident(entry)?;
    if linear.op != Ops::Linear {
        return Err("GPU renderer requires lowered LINEAR UOps".into());
    }
    let mut params = BTreeMap::new();
    let mut values = HashMap::<u64, String>::new();
    let mut ranges = HashSet::new();
    let mut body = String::new();
    let mut declarations = String::new();
    let shared_bytes = 0usize;
    let core_signature = crate::codegen::signature(linear)?;
    let buffer_slots = core_signature
        .buffers
        .iter()
        .map(|(slot, u)| (u.id, *slot))
        .collect::<HashMap<_, _>>();
    for (slot, _) in &core_signature.buffers {
        params.insert(*slot, format!("p{slot}"));
    }
    for u in &linear.src {
        if let Arg::Slot { index, .. } = u.arg {
            if u.op == Ops::Param {
                params.insert(index, format!("p{index}"));
            }
        }
    }
    let mut source = match dialect {
        Dialect::Metal => "#include <metal_stdlib>\nusing namespace metal;\n".to_owned(),
        Dialect::OpenCl => String::new(),
    };
    source
        .push_str(
            "inline long chibi_div(long a,long b){long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\ninline long chibi_mod(long a,long b){long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\n",
        );
    {}
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
            Dialect::OpenCl => format!("__global float* {name}"),
        })
        .collect::<Vec<_>>();
    if dialect == Dialect::Metal {
        signature
            .extend([
                "uint3 gid [[thread_position_in_grid]]".into(),
                "uint3 lid [[thread_position_in_threadgroup]]".into(),
                "uint3 group [[threadgroup_position_in_grid]]".into(),
            ]);
    }
    for u in &linear.src {
        let name = format!("v{}", u.id);
        let get = |i: usize| -> Result<String, String> {
            let s = u
                .src
                .get(i)
                .ok_or_else(|| format!("missing {:?} source {i}", u.op))?;
            values
                .get(&s.id)
                .cloned()
                .ok_or_else(|| {
                    format!("GPU UOp {:?} source {} not linearized", u.op, s.id)
                })
        };
        let expr = match u.op {
            Ops::Const => {
                match u.arg {
                    Arg::Float(bits) => {
                        let bits = (bits.value() as f32).to_bits();
                        match dialect {
                            Dialect::Metal => format!("as_type<float>((uint){bits})"),
                            Dialect::OpenCl => format!("as_float((uint){bits})"),
                        }
                    }
                    Arg::I32(x) => format!("{x}"),
                    Arg::Int(x) => format!("{x}L"),
                    Arg::Bool(b) => if b { "true" } else { "false" }.into(),
                    _ => return Err("unsupported GPU constant".into()),
                }
            }
            Ops::Alloc => {
                if let Arg::Allocation { dtype, size, .. } = u.arg {
                    if size != 1 {
                        return Err(
                            "GPU scratch arrays require the parallel reduction stage"
                                .into(),
                        );
                    }
                    let address = "";
                    declarations
                        .push_str(&format!("{address}{} {name}[{size}];\n", ty(dtype)));
                    values.insert(u.id, name);
                    continue;
                } else {
                    return Err("GPU allocation missing layout".into());
                }
            }
            Ops::Param | Ops::Buffer => {
                format!(
                    "p{}", buffer_slots.get(& u.id).ok_or("GPU buffer missing slot") ?
                )
            }
            Ops::Index => {
                if let Arg::DType(DType::F32x4) = u.arg {
                    let address = if dialect == Dialect::Metal {
                        "device "
                    } else {
                        "__global "
                    };
                    format!("((({address}float4*){})+({}))", get(0) ?, get(1) ?)
                } else {
                    format!("({}+({}))", get(0) ?, get(1) ?)
                }
            }
            Ops::Load => {
                {}
                let index = &u.src[0];
                let ptr = get(0)?;
                let cast = match u.arg {
                    Arg::DType(dt) => format!("{}*", ty(dt)),
                    _ => "float*".into(),
                };
                let addr = if dialect == Dialect::Metal {
                    "device "
                } else {
                    "__global "
                };
                let typed = if matches!(u.arg, Arg::DType(_)) {
                    format!("(({addr}{cast}){ptr})")
                } else {
                    ptr
                };
                if index.op == Ops::Index && index.src.len() > 2
                    && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU load mask")?;
                    format!("({valid}?*{typed}:0)")
                } else {
                    format!("*{typed}")
                }
            }
            Ops::Store => {
                let ptr = get(0)?;
                let value = get(1)?;
                {}
                let index = &u.src[0];
                if index.op == Ops::Index && index.src.len() > 2
                    && index.src[2] != UOp::bool(true)
                {
                    let valid = values
                        .get(&index.src[2].id)
                        .ok_or("missing GPU store mask")?;
                    body.push_str(&format!("if({valid}) *{ptr}={value};\n"));
                } else {
                    body.push_str(&format!("*{ptr}={value};\n"));
                }
                String::new()
            }
            Ops::Range => {
                let (id, axis) = if let Arg::Range { id, axis } = u.arg {
                    (id, axis)
                } else {
                    return Err("range missing axis".into());
                };
                match axis {
                    Axis::Global => {
                        let dim = id % 3;
                        let hw = match (dialect, axis) {
                            (Dialect::Metal, Axis::Global) => format!("gid[{}]", dim),
                            (Dialect::Metal, _) => format!("lid[{}]", dim),
                            (Dialect::OpenCl, Axis::Global) => {
                                format!("get_global_id({dim})")
                            }
                            _ => format!("get_local_id({dim})"),
                        };
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!("{name}={hw};\n"));
                    }
                    _ => {
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(
                            &format!(
                                "for({name}={};{name}<{};{name}++){{\n", get(0) ?, get(1) ?
                            ),
                        );
                        ranges.insert(u.id);
                    }
                }
                values.insert(u.id, name);
                continue;
            }
            Ops::End => {
                if u.src.last().is_some_and(|r| ranges.contains(&r.id)) {
                    body.push_str("}\n");
                }
                String::new()
            }
            Ops::Special => {
                let text = if let Arg::Text(ref t) = u.arg {
                    t.as_str()
                } else {
                    return Err("SPECIAL requires gid/lid/group and dimension".into());
                };
                let (kind, dim) = text.split_at(text.len().saturating_sub(1));
                let dim = dim.parse::<usize>().map_err(|_| "invalid SPECIAL dimension")?;
                if dim > 2 {
                    return Err("SPECIAL dimension outside grid".into());
                }
                match (dialect, kind) {
                    (Dialect::Metal, "gid") => format!("gid[{dim}]"),
                    (Dialect::Metal, "lid") => format!("lid[{dim}]"),
                    (Dialect::Metal, "group") => format!("group[{dim}]"),
                    (Dialect::OpenCl, "gid") => format!("get_global_id({dim})"),
                    (Dialect::OpenCl, "lid") => format!("get_local_id({dim})"),
                    (Dialect::OpenCl, "group") => format!("get_group_id({dim})"),
                    _ => return Err("invalid SPECIAL builtin".into()),
                }
            }
            Ops::Div if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", get(0) ?, get(1) ?)
            }
            Ops::Mod => format!("chibi_mod({},{})", get(0) ?, get(1) ?),
            Ops::Add
            | Ops::Sub
            | Ops::Mul
            | Ops::Div
            | Ops::Lt
            | Ops::Eq
            | Ops::And
            | Ops::Or => {
                let op = match u.op {
                    Ops::Add => "+",
                    Ops::Sub => "-",
                    Ops::Mul => "*",
                    Ops::Div => "/",
                    Ops::Mod => "%",
                    Ops::Lt => "<",
                    Ops::Eq => "==",
                    Ops::And => "&&",
                    _ => "||",
                };
                format!("({} {op} {})", get(0) ?, get(1) ?)
            }
            Ops::Max => format!("max({},{})", get(0) ?, get(1) ?),
            Ops::Neg => format!("(-{})", get(0) ?),
            Ops::Exp
            | Ops::Log
            | Ops::Sin
            | Ops::Cos
            | Ops::Sqrt
            | Ops::Exp2
            | Ops::Log2 => {
                let fun = match u.op {
                    Ops::Exp => "exp",
                    Ops::Log => "log",
                    Ops::Sin => "sin",
                    Ops::Cos => "cos",
                    Ops::Sqrt => "sqrt",
                    Ops::Exp2 => "exp2",
                    _ => "log2",
                };
                format!("{fun}({})", get(0) ?)
            }
            Ops::Where => format!("({}?{}:{})", get(0) ?, get(1) ?, get(2) ?),
            Ops::Cast => format!("(({}){})", ty(u.dtype()), get(0) ?),
            Ops::After => get(0)?,
            Ops::Sink | Ops::Linear => String::new(),
            _ => return Err(format!("unsupported lowered GPU UOp {:?}", u.op)),
        };
        if !expr.is_empty() {
            if matches!(u.op, Ops::Index | Ops::Param | Ops::Buffer | Ops::After) {
                values.insert(u.id, expr);
            } else {
                declarations.push_str(&format!("{} {name};\n", ty(u.dtype())));
                body.push_str(&format!("{name}={expr};\n"));
                values.insert(u.id, name);
            }
        }
    }
    let mut declared = HashSet::new();
    let declarations = declarations
        .lines()
        .filter(|line| declared.insert(*line))
        .collect::<Vec<_>>()
        .join("\n");
    source
        .push_str(
            &format!(
                "{} void {entry}({}){{\n{declarations}{body}}}\n", match dialect {
                Dialect::Metal => "kernel", Dialect::OpenCl => "__kernel", }, signature
                .join(",")
            ),
        );
    Ok(Kernel {
        source,
        entry: entry.into(),
        launch,
        shared_bytes,
    })
}
```

`src/uir.rs` の `fn dtype`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn dtype"
fn dtype(s: &str) -> Result<DType> {
    [
        DType::F32,
        DType::F32x4,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::Ptr,
        DType::Void,
    ]
    .into_iter()
    .find(|d| format!("{d:?}") == s)
    .ok_or("unknown dtype".into())
}
```

`src/uir.rs` の `fn op`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn op"
fn op(s: &str) -> Result<Ops> {
    [
        Ops::Invalid,
        Ops::Const,
        Ops::Special,
        Ops::Buffer,
        Ops::Alloc,
        Ops::Param,
        Ops::Cast,
        Ops::Bitcast,
        Ops::Shl,
        Ops::Shr,
        Ops::Xor,
        Ops::Add,
        Ops::Sub,
        Ops::Mul,
        Ops::Div,
        Ops::Mod,
        Ops::Neg,
        Ops::Exp,
        Ops::Log,
        Ops::Sin,
        Ops::Cos,
        Ops::Sqrt,
        Ops::Exp2,
        Ops::Log2,
        Ops::Max,
        Ops::Lt,
        Ops::Eq,
        Ops::And,
        Ops::Or,
        Ops::Where,
        Ops::Reshape,
        Ops::Permute,
        Ops::Expand,
        Ops::Shrink,
        Ops::Pad,
        Ops::Flip,
        Ops::Stack,
        Ops::Gather,
        Ops::ScatterAdd,
        Ops::Reduce,
        Ops::Detach,
        Ops::Stage,
        Ops::Range,
        Ops::Index,
        Ops::Load,
        Ops::Store,
        Ops::End,
        Ops::After,
        Ops::Call,
        Ops::Sink,
        Ops::Linear,
        Ops::Program,
        Ops::Source,
        Ops::Binary,
        Ops::CustomFunction,
    ]
    .into_iter()
    .find(|o| format!("{o:?}") == s)
    .ok_or("unknown operation".into())
}
```

`src/uop.rs` の `pub enum DType`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum DType"
pub enum DType {
    F32,
    I32,
    I64,
    Bool,
    Ptr,
    Void,
    F32x4,
}
```

`src/uop.rs` の `pub enum Ops`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Ops"
pub enum Ops {
    Invalid,
    Const,
    Buffer,
    Alloc,
    Param,
    Cast,
    Bitcast,
    Shl,
    Shr,
    Xor,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Exp2,
    Log2,
    Max,
    Lt,
    Eq,
    And,
    Or,
    Where,
    Reshape,
    Permute,
    Expand,
    Shrink,
    Pad,
    Flip,
    Stack,
    Gather,
    ScatterAdd,
    Reduce,
    Detach,
    Stage,
    Range,
    Index,
    Load,
    Store,
    End,
    After,
    Call,
    Sink,
    Linear,
    Program,
    Source,
    Binary,
    CustomFunction,
    Special,
}
```

`src/uop.rs` の `impl UOp` 内 の `fn infer_dtype`を置き換える。

```rust
// file="src/uop.rs" mode="replace" within="impl UOp" item="fn infer_dtype"
    fn infer_dtype(&self) -> DType {
        match self.op {
            Ops::Const => match self.arg {
                Arg::Float(_) => DType::F32,
                Arg::I32(_) => DType::I32,
                Arg::Int(_) => DType::I64,
                Arg::Bool(_) => DType::Bool,
                _ => DType::Void,
            },
            Ops::Param => match self.arg {
                Arg::ScalarParam { dtype, .. } => dtype,
                _ => DType::Ptr,
            },
            Ops::Buffer | Ops::Alloc => DType::Ptr,
            Ops::Index => {
                if self.src.first().is_some_and(|s| s.op == Ops::Call) {
                    self.src[0].dtype()
                } else {
                    DType::Ptr
                }
            }
            Ops::Range | Ops::Special => DType::I64,
            Ops::Lt | Ops::Eq => DType::Bool,
            Ops::Cast | Ops::Bitcast => {
                if let Arg::DType(dt) = self.arg {
                    dt
                } else {
                    DType::Void
                }
            }
            Ops::CustomFunction => {
                if let Arg::Function { dtype, .. } = self.arg { dtype } else { DType::Void }
            }
            Ops::Load => match self.arg {
                Arg::DType(dt) => dt,
                _ => DType::F32,
            },
            Ops::Gather | Ops::ScatterAdd => DType::F32,
            Ops::Call => self.src.first().map_or(DType::Void, |x| x.dtype()),
            Ops::Where => self.src.get(1).map_or(DType::Void, |x| x.dtype()),
            Ops::Store
            | Ops::End
            | Ops::Sink
            | Ops::Linear
            | Ops::Program
            | Ops::Source
            | Ops::Binary
            | Ops::Invalid => DType::Void,
            _ => self.src.first().map_or(DType::Void, |x| {
                if x.dtype() == DType::Ptr {
                    DType::F32
                } else {
                    x.dtype()
                }
            }),
        }
    }
```

`tests/vector_cpu.rs`を保存する。

```rust
// file="tests/vector_cpu.rs" mode="write"
use chibigrad::{Arg, Axis, DType, Ops, UOp, codegen, cpu::CpuProgram};
#[test]
fn vector4_load_add_store_execute_native_cpu() {
    let pointer = |index| {
        UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index,
                shape: vec![16],
            },
        )
    };
    let r = UOp::range(UOp::int(4), 0, Axis::Loop);
    let index = |p: UOp| {
        UOp::new(
            Ops::Index,
            vec![p, r.clone(), UOp::bool(true)],
            Arg::DType(DType::F32x4),
        )
    };
    let load = |p: UOp| UOp::new(Ops::Load, vec![index(p)], Arg::DType(DType::F32x4));
    let result = load(pointer(0)).add(&load(pointer(1)));
    let store = index(pointer(2)).store(result);
    let end = UOp::new(Ops::End, vec![store.clone(), r.clone()], Arg::None);
    let source = codegen::render(&UOp::linear(vec![r, store, end]), "vector4_entry").unwrap();
    let program = CpuProgram::compile(&source, "vector4_entry").unwrap();
    let a = (0..16).map(|x| x as f32).collect::<Vec<_>>();
    let b = (0..16).map(|x| x as f32 * 3.).collect::<Vec<_>>();
    let mut out = vec![0.0f32; 16];
    let mut args = (a.as_ptr() as u64).to_ne_bytes().to_vec();
    args.extend((b.as_ptr() as u64).to_ne_bytes());
    args.extend((out.as_mut_ptr() as u64).to_ne_bytes());
    unsafe { program.run(&mut args) }.unwrap();
    assert_eq!(out, (0..16).map(|x| x as f32 * 4.).collect::<Vec<_>>());
}
```

```sh
# check step=17.10
cargo test --offline --test vector_cpu
```

CPUの16要素の加算結果が参照値と一致する。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 17-001 `i=outer*tile+inner` への置換を kernel IR に追加する。

<!-- todo:17-001 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-002 outer の範囲を切上げ、元範囲の guard を作る。

<!-- todo:17-002 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-003 無効な要素の LOAD と STORE を guard の内側に置く。

<!-- todo:17-003 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-004 tile=0、負値、添字の overflow を起動前に拒否する。

<!-- todo:17-004 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-005 split 前と split 後を同じ入力で実行する。

<!-- todo:17-005 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-006 独立した出力軸の順序を入れ替える変換を追加する。

<!-- todo:17-006 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-007 reduction 軸の再順序化は初版では明示的に禁止する。

<!-- todo:17-007 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-008 小さい inner RANGE を定数座標ごとに複製する。

<!-- todo:17-008 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-009 RANGE とローカル変数を複製先の scope ごとに作り直す。

<!-- todo:17-009 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-010 展開後にも tail guard と累積の初期化位置を維持する。

<!-- todo:17-010 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-011 UPCAST 軸を固定個数の scalar 値へ展開する。

<!-- todo:17-011 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-012 連続した width=4 の f32 を vector 型へまとめる変換を別に追加する。

<!-- todo:17-012 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-013 vector LOAD/STORE の合法性を調べ、misalignment には安全な方法を使う。

<!-- todo:17-013 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-014 端数は scalar loop に残し、full vector の範囲外 LOAD を避ける。

<!-- todo:17-014 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-015 vector 非対応 backend は scalar へ戻す。

<!-- todo:17-015 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-016 CPU runtime に固定数の worker と join を追加する。

<!-- todo:17-016 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-017 出力区間を重複と欠落がないように分割する。

<!-- todo:17-017 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-018 区間の begin/end と引数表を worker ごとに持つ。

<!-- todo:17-018 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-019 worker が同じ compiler やキャッシュへ同時に書かないよう、compile を先に完了する。

<!-- todo:17-019 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-020 thread 数 1、2、3、出力数より多い場合を比較する。

<!-- todo:17-020 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-021 backend ごとに vector 幅、CPU thread、LOCAL、shared memory の可否を明示する。

<!-- todo:17-021 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-022 未対応変換を黙って別の意味で実行しない。

<!-- todo:17-022 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-023 split、unroll、vector、thread を単独で検証した後に組み合わせる。

<!-- todo:17-023 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-024 compile、転送、実行、同期を別に測れる計測点だけを追加する。

<!-- todo:17-024 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-025 split と unroll の後にも scope と buffer 境界の verifier が通る。

<!-- todo:17-025 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-026 未対応 vector の fallback と CPU thread 数 1 が基準版と一致する。

<!-- todo:17-026 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-027 出力ごとの所有区間が明確で、reduction の累積を共有しない。

<!-- todo:17-027 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 17-028 GPU local と CPU worker の capability を独立して表示する。

<!-- todo:17-028 -->

関連する実装と確認は本章の `17.01`, `17.02`, `17.03`, `17.04`, `17.05`, `17.06`, `17.07`, `17.08`, `17.09`, `17.10` にある。`tests/vector_cpu.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
