# 15. Metalの転送とカーネル実行を追加する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 15.01 Metal デバイスと共有バッファ

Metal を動的に読み込み、所有権を持つバッファへアップロードして読み戻す。範囲検査とデバイス寿命を先に確認する。

`Cargo.toml`を追記する。

```toml
# file="Cargo.toml" mode="append"

[features]
default=["native"]
native=[]
metal=["native"]
```

`src/backends/dynamic.rs`を保存する。

```rust
// file="src/backends/dynamic.rs" mode="write"
use std::{
    collections::HashMap,
    ffi::{CString, c_char, c_int, c_void},
    sync::{Mutex, OnceLock},
};
#[cfg_attr(not(target_os = "macos"), link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
pub struct Library(*mut c_void);
static LIBRARIES: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
impl Library {
    pub fn open(path: &str) -> Result<Self, String> {
        let mut libraries = LIBRARIES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| "library cache lock poisoned")?;
        if let Some(&handle) = libraries.get(path) {
            return Ok(Self(handle as *mut c_void));
        }
        let name = CString::new(path).map_err(|_| "invalid library path")?;
        let ptr = unsafe { dlopen(name.as_ptr(), 2) };
        if ptr.is_null() {
            Err(format!("library unavailable: {path}"))
        } else {
            libraries.insert(path.into(), ptr as usize);
            Ok(Self(ptr))
        }
    }
    pub fn symbol(&self, name: &str) -> Result<*mut c_void, String> {
        let name_c = CString::new(name).map_err(|_| "invalid symbol")?;
        let ptr = unsafe { dlsym(self.0, name_c.as_ptr()) };
        if ptr.is_null() {
            Err(format!("foreign symbol unavailable: {name}"))
        } else {
            Ok(ptr)
        }
    }
}
```

`src/backends/metal.rs`を保存する。

```rust
// file="src/backends/metal.rs" mode="write"
use super::dynamic::Library;
use crate::gpu::{Capabilities, Kernel, Launch, span};
use std::{
    ffi::{CString, c_char, c_void},
    rc::Rc,
    time::{Duration, Instant},
};
type Id = usize;
#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    width: usize,
    height: usize,
    depth: usize,
}
impl From<[usize; 3]> for Size {
    fn from(v: [usize; 3]) -> Self {
        Self {
            width: v[0],
            height: v[1],
            depth: v[2],
        }
    }
}
struct Api {
    _metal: Library,
    _objc: Library,
    _foundation: Library,
    _graphics: Library,
    msg: *mut c_void,
    selector: unsafe extern "C" fn(*const c_char) -> Id,
    class: unsafe extern "C" fn(*const c_char) -> Id,
}
macro_rules! send {
    ($a:expr, $o:expr, $s:literal, $r:ty $(, $v:expr => $t:ty)*) => {{
        let a = &$a;
        let selector = unsafe { (a.selector)(concat!($s,"\0").as_ptr().cast()) };
        let call: unsafe extern "C" fn(Id, Id $(,$t)*) -> $r = unsafe { std::mem::transmute(a.msg) };
        unsafe { call($o, selector $(,$v)*) }
    }};
}
impl Api {
    fn load() -> Result<Rc<Self>, String> {
        let objc = Library::open("/usr/lib/libobjc.A.dylib")?;
        let metal = Library::open("/System/Library/Frameworks/Metal.framework/Metal")?;
        let foundation =
            Library::open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
        let graphics =
            Library::open("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
        Ok(Rc::new(Self {
            msg: objc.symbol("objc_msgSend")?,
            selector: unsafe { std::mem::transmute(objc.symbol("sel_registerName")?) },
            class: unsafe { std::mem::transmute(objc.symbol("objc_getClass")?) },
            _metal: metal,
            _objc: objc,
            _foundation: foundation,
            _graphics: graphics,
        }))
    }
    fn class(&self, name: &str) -> Id {
        let s = CString::new(name).unwrap();
        unsafe { (self.class)(s.as_ptr()) }
    }
    fn text(&self, o: Id) -> String {
        if o == 0 {
            return String::new();
        }
        let p = send!(self, o, "UTF8String", *const c_char);
        if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    }
}
struct Object {
    ptr: Id,
}
impl Object {
    fn new(ptr: Id, name: &str) -> Result<Self, String> {
        if ptr == 0 {
            Err(format!("Metal {name} returned nil"))
        } else {
            Ok(Self { ptr })
        }
    }
}
struct Owned {
    obj: Object,
    api: Rc<Api>,
}
impl Owned {
    fn new(api: &Rc<Api>, ptr: Id, name: &str) -> Result<Self, String> {
        Ok(Self {
            obj: Object::new(ptr, name)?,
            api: api.clone(),
        })
    }
    fn id(&self) -> Id {
        self.obj.ptr
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        send!(self.api, self.id(), "release", ());
    }
}
#[derive(Clone)]
pub struct MetalDevice {
    inner: Rc<DeviceInner>,
}
struct DeviceInner {
    api: Rc<Api>,
    device: Owned,
    caps: Capabilities,
}
#[derive(Clone)]
pub struct MetalBuffer {
    inner: Rc<BufferInner>,
}
struct BufferInner {
    native: Owned,
    size: usize,
    device: Rc<DeviceInner>,
}
impl MetalDevice {
    pub fn open() -> Result<Self, String> {
        let _initializing = super::INITIALIZE
            .lock()
            .map_err(|_| "device initialization poisoned")?;
        let api = Api::load()?;
        let create: unsafe extern "C" fn() -> Id =
            unsafe { std::mem::transmute(api._metal.symbol("MTLCreateSystemDefaultDevice")?) };
        let device = Owned::new(&api, unsafe { create() }, "device")?;
        let mut caps = Capabilities {
            name: api.text(send!(api, device.id(), "name", Id)),
            max_threads: send!(api, device.id(), "maxThreadsPerThreadgroup", Size).width,
            local_memory: send!(api, device.id(), "maxThreadgroupMemoryLength", usize),
            ..Capabilities::default()
        };
        for family in 1..=9 {
            if send!(api,device.id(),"supportsFamily:",bool,1000+family=>usize) {
                caps.families.push(format!("Apple{family}"));
            }
        }
        Ok(Self {
            inner: Rc::new(DeviceInner { api, device, caps }),
        })
    }
    pub fn capabilities(&self) -> &Capabilities {
        &self.inner.caps
    }
    pub fn allocate(&self, size: usize) -> Result<MetalBuffer, String> {
        let i = &self.inner;
        let max = send!(i.api, i.device.id(), "maxBufferLength", usize);
        if size > max {
            return Err("Metal allocation exceeds device limit".into());
        }
        let native = Owned::new(
            &i.api,
            send!(i.api,i.device.id(),"newBufferWithLength:options:",Id,size.max(8)=>usize,0=>usize),
            "buffer",
        )?;
        Ok(MetalBuffer {
            inner: Rc::new(BufferInner {
                native,
                size,
                device: i.clone(),
            }),
        })
    }
}
impl MetalBuffer {
    pub fn len(&self) -> usize {
        self.inner.size
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        span(self.len(), offset, bytes.len())?;
        let p = send!(
            self.inner.device.api,
            self.inner.native.id(),
            "contents",
            *mut u8
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(offset), bytes.len());
        }
        Ok(())
    }
    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, String> {
        span(self.len(), offset, length)?;
        let p = send!(
            self.inner.device.api,
            self.inner.native.id(),
            "contents",
            *mut u8
        );
        Ok(unsafe { std::slice::from_raw_parts(p.add(offset), length) }.to_vec())
    }
    pub fn device_address(&self) -> u64 {
        send!(
            self.inner.device.api,
            self.inner.native.id(),
            "gpuAddress",
            u64
        )
    }
}
```

`src/backends/mod.rs`を保存する。

```rust
// file="src/backends/mod.rs" mode="write"
pub(crate) mod dynamic;
pub(crate) static INITIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
```

`src/gpu.rs`を保存する。

```rust
// file="src/gpu.rs" mode="write"
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub groups: [usize; 3],
    pub local: [usize; 3],
}
impl Launch {
    pub fn validate(&self, max_threads: usize) -> Result<(), String> {
        let n = self
            .local
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or("local size overflow")?;
        if n == 0 || n > max_threads || self.groups.contains(&0) {
            return Err("invalid launch dimensions".into());
        }
        for i in 0..3 {
            self.groups[i]
                .checked_mul(self.local[i])
                .ok_or("global size overflow")?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Kernel {
    pub source: String,
    pub entry: String,
    pub launch: Launch,
    pub shared_bytes: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    pub max_threads: usize,
    pub local_memory: usize,
    pub indirect_commands: bool,
    pub families: Vec<String>,
    pub extensions: Vec<String>,
}
pub(crate) fn span(size: usize, offset: usize, length: usize) -> Result<(), String> {
    if offset.checked_add(length).is_none_or(|n| n > size) {
        Err("buffer access outside allocation".into())
    } else {
        Ok(())
    }
}
```

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod gpu;


#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub mod backends;
```

`tests/metal_buffer.rs`を保存する。

```rust
// file="tests/metal_buffer.rs" mode="write"
#![cfg(all(feature = "metal", target_os = "macos"))]
use chibigrad::backends::metal::MetalDevice;
#[test]
fn native_buffer_owns_device_and_checks_spans() {
    let device = MetalDevice::open().unwrap();
    let buffer = device.allocate(16).unwrap();
    buffer.write(0, &[1, 2, 3, 4]).unwrap();
    drop(device);
    assert_eq!(buffer.read(0, 4).unwrap(), vec![1, 2, 3, 4]);
    assert!(buffer.write(15, &[1, 2]).is_err());
    assert!(buffer.read(usize::MAX, 2).is_err());
}
```

```sh
# check step=15.01 requires=metal
cargo test --features metal --test metal_buffer -- --nocapture
```

4 バイトが一致し、範囲外アクセスは失敗する。

## 15.02 Metal カーネルの直接実行

Rust ホストからソースをコンパイルし、引数を束縛して送信と完了待ちを行う。同じコマンドを再生し、バッファの差し替えとイベントの順序も確かめる。

`src/backends/metal.rs`を保存する。

```rust
// file="src/backends/metal.rs" mode="write"
use super::dynamic::Library;
use crate::gpu::{Capabilities, Kernel, Launch, span};
use std::{
    ffi::{CString, c_char, c_void},
    rc::Rc,
    time::{Duration, Instant},
};
type Id = usize;
#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    width: usize,
    height: usize,
    depth: usize,
}
impl From<[usize; 3]> for Size {
    fn from(v: [usize; 3]) -> Self {
        Self {
            width: v[0],
            height: v[1],
            depth: v[2],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Range {
    location: usize,
    length: usize,
}
struct Api {
    _metal: Library,
    _objc: Library,
    _foundation: Library,
    _graphics: Library,
    msg: *mut c_void,
    selector: unsafe extern "C" fn(*const c_char) -> Id,
    class: unsafe extern "C" fn(*const c_char) -> Id,
}
macro_rules! send {
    ($a:expr, $o:expr, $s:literal, $r:ty $(, $v:expr => $t:ty)*) => {{
        let a = &$a;
        let selector = unsafe { (a.selector)(concat!($s,"\0").as_ptr().cast()) };
        let call: unsafe extern "C" fn(Id, Id $(,$t)*) -> $r = unsafe { std::mem::transmute(a.msg) };
        unsafe { call($o, selector $(,$v)*) }
    }};
}
impl Api {
    fn load() -> Result<Rc<Self>, String> {
        let objc = Library::open("/usr/lib/libobjc.A.dylib")?;
        let metal = Library::open("/System/Library/Frameworks/Metal.framework/Metal")?;
        let foundation =
            Library::open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
        let graphics =
            Library::open("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
        Ok(Rc::new(Self {
            msg: objc.symbol("objc_msgSend")?,
            selector: unsafe { std::mem::transmute(objc.symbol("sel_registerName")?) },
            class: unsafe { std::mem::transmute(objc.symbol("objc_getClass")?) },
            _metal: metal,
            _objc: objc,
            _foundation: foundation,
            _graphics: graphics,
        }))
    }
    fn class(&self, name: &str) -> Id {
        let s = CString::new(name).unwrap();
        unsafe { (self.class)(s.as_ptr()) }
    }
    fn string(self: &Rc<Self>, s: &str) -> Result<Owned, String> {
        let s = CString::new(s).map_err(|_| "NUL in Metal source")?;
        let o = send!(self, self.class("NSString"), "alloc", Id);
        let o = send!(self,o,"initWithUTF8String:",Id,s.as_ptr()=>*const c_char);
        Owned::new(self, o, "NSString")
    }
    fn text(&self, o: Id) -> String {
        if o == 0 {
            return String::new();
        }
        let p = send!(self, o, "UTF8String", *const c_char);
        if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    }
}
struct Object {
    ptr: Id,
}
impl Object {
    fn new(ptr: Id, name: &str) -> Result<Self, String> {
        if ptr == 0 {
            Err(format!("Metal {name} returned nil"))
        } else {
            Ok(Self { ptr })
        }
    }
}
struct Owned {
    obj: Object,
    api: Rc<Api>,
}
impl Owned {
    fn new(api: &Rc<Api>, ptr: Id, name: &str) -> Result<Self, String> {
        Ok(Self {
            obj: Object::new(ptr, name)?,
            api: api.clone(),
        })
    }
    fn id(&self) -> Id {
        self.obj.ptr
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        send!(self.api, self.id(), "release", ());
    }
}
#[derive(Clone)]
pub struct MetalDevice {
    inner: Rc<DeviceInner>,
}
struct DeviceInner {
    api: Rc<Api>,
    device: Owned,
    queue: Owned,
    event: Owned,
    caps: Capabilities,
    submitted: std::cell::Cell<u64>,
}
#[derive(Clone)]
pub struct MetalBuffer {
    inner: Rc<BufferInner>,
}
struct BufferInner {
    native: Owned,
    size: usize,
    device: Rc<DeviceInner>,
}
#[derive(Clone)]
pub struct MetalProgram {
    inner: Rc<ProgramInner>,
}
struct ProgramInner {
    native: Owned,
    device: Rc<DeviceInner>,
    max_threads: usize,
}
pub struct MetalCommandBuffer {
    icb: Owned,
    device: Rc<DeviceInner>,
    launches: Vec<Launch>,
    bindings: Vec<Vec<MetalBuffer>>,
    programs: Vec<MetalProgram>,
    in_flight: Option<Owned>,
    last_signal: u64,
}
impl MetalDevice {
    pub fn open() -> Result<Self, String> {
        let _initializing = super::INITIALIZE
            .lock()
            .map_err(|_| "device initialization poisoned")?;
        let api = Api::load()?;
        let create: unsafe extern "C" fn() -> Id =
            unsafe { std::mem::transmute(api._metal.symbol("MTLCreateSystemDefaultDevice")?) };
        let device = Owned::new(&api, unsafe { create() }, "device")?;
        let queue = Owned::new(
            &api,
            send!(api, device.id(), "newCommandQueue", Id),
            "queue",
        )?;
        let event = Owned::new(
            &api,
            send!(api, device.id(), "newSharedEvent", Id),
            "shared event",
        )?;
        let mut caps = Capabilities {
            name: api.text(send!(api, device.id(), "name", Id)),
            max_threads: send!(api, device.id(), "maxThreadsPerThreadgroup", Size).width,
            local_memory: send!(api, device.id(), "maxThreadgroupMemoryLength", usize),
            indirect_commands: true,
            ..Capabilities::default()
        };
        for family in 1..=9 {
            if send!(api,device.id(),"supportsFamily:",bool,1000+family=>usize) {
                caps.families.push(format!("Apple{family}"));
            }
        }
        Ok(Self {
            inner: Rc::new(DeviceInner {
                api,
                device,
                queue,
                event,
                caps,
                submitted: std::cell::Cell::new(0),
            }),
        })
    }
    pub fn capabilities(&self) -> &Capabilities {
        &self.inner.caps
    }
    pub fn allocate(&self, size: usize) -> Result<MetalBuffer, String> {
        let i = &self.inner;
        let max = send!(i.api, i.device.id(), "maxBufferLength", usize);
        if size > max {
            return Err("Metal allocation exceeds device limit".into());
        }
        let native = Owned::new(
            &i.api,
            send!(i.api,i.device.id(),"newBufferWithLength:options:",Id,size.max(8)=>usize,0=>usize),
            "buffer",
        )?;
        Ok(MetalBuffer {
            inner: Rc::new(BufferInner {
                native,
                size,
                device: i.clone(),
            }),
        })
    }
    pub fn compile(&self, kernel: &Kernel) -> Result<MetalProgram, String> {
        let i = &self.inner;
        let source = i.api.string(&kernel.source)?;
        let entry = i.api.string(&kernel.entry)?;
        let mut error = 0usize;
        let library=Owned::new(&i.api,send!(i.api,i.device.id(),"newLibraryWithSource:options:error:",Id,source.id()=>Id,0=>Id,&mut error=>*mut Id),"library").map_err(|e|format!("{e}: {}", i.api.text(send!(i.api,error,"localizedDescription",Id))))?;
        let function = Owned::new(
            &i.api,
            send!(i.api,library.id(),"newFunctionWithName:",Id,entry.id()=>Id),
            "function",
        )?;
        let descriptor = Owned::new(
            &i.api,
            send!(
                i.api,
                i.api.class("MTLComputePipelineDescriptor"),
                "new",
                Id
            ),
            "pipeline descriptor",
        )?;
        send!(i.api,descriptor.id(),"setComputeFunction:",(),function.id()=>Id);
        send!(i.api,descriptor.id(),"setSupportIndirectCommandBuffers:",(),true=>bool);
        let pipeline = Owned::new(
            &i.api,
            send!(i.api,i.device.id(),"newComputePipelineStateWithDescriptor:options:reflection:error:",Id,descriptor.id()=>Id,0=>usize,std::ptr::null_mut::<Id>()=>*mut Id,&mut error=>*mut Id),
            "pipeline",
        )?;
        let max_threads = send!(i.api, pipeline.id(), "maxTotalThreadsPerThreadgroup", usize);
        kernel.launch.validate(max_threads)?;
        if kernel.shared_bytes > i.caps.local_memory
            || send!(i.api, pipeline.id(), "staticThreadgroupMemoryLength", usize)
                > i.caps.local_memory
        {
            return Err("Metal local memory limit".into());
        }
        Ok(MetalProgram {
            inner: Rc::new(ProgramInner {
                native: pipeline,
                device: i.clone(),
                max_threads,
            }),
        })
    }
    pub fn command_buffer(
        &self,
        commands: &[(MetalProgram, Launch, Vec<MetalBuffer>)],
    ) -> Result<MetalCommandBuffer, String> {
        let i = &self.inner;
        let desc = Owned::new(
            &i.api,
            send!(
                i.api,
                i.api.class("MTLIndirectCommandBufferDescriptor"),
                "new",
                Id
            ),
            "ICB descriptor",
        )?;
        send!(i.api,desc.id(),"setCommandTypes:",(),32=>usize);
        send!(i.api,desc.id(),"setInheritBuffers:",(),false=>bool);
        let max_buffers = commands.iter().map(|c| c.2.len()).max().unwrap_or(1);
        send!(i.api,desc.id(),"setMaxKernelBufferBindCount:",(),max_buffers=>usize);
        let icb = Owned::new(
            &i.api,
            send!(i.api,i.device.id(),"newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",Id,desc.id()=>Id,commands.len().max(1)=>usize,0=>usize),
            "indirect command buffer",
        )?;
        let mut bindings = vec![];
        let mut programs = vec![];
        let mut launches = vec![];
        for (idx, (program, launch, buffers)) in commands.iter().enumerate() {
            if !Rc::ptr_eq(i, &program.inner.device) {
                return Err("Metal pipeline belongs to another device".into());
            }
            launch.validate(program.inner.max_threads)?;
            let cmd = send!(i.api,icb.id(),"indirectComputeCommandAtIndex:",Id,idx=>usize);
            send!(i.api,cmd,"setComputePipelineState:",(),program.inner.native.id()=>Id);
            for (j, b) in buffers.iter().enumerate() {
                if !Rc::ptr_eq(i, &b.inner.device) {
                    return Err("Metal buffer belongs to another device".into());
                }
                send!(i.api,cmd,"setKernelBuffer:offset:atIndex:",(),b.inner.native.id()=>Id,0=>usize,j=>usize);
            }
            send!(i.api,cmd,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",(),Size::from(launch.groups)=>Size,Size::from(launch.local)=>Size);
            send!(i.api, cmd, "setBarrier", ());
            bindings.push(buffers.clone());
            programs.push(program.clone());
            launches.push(launch.clone());
        }
        Ok(MetalCommandBuffer {
            icb,
            device: i.clone(),
            launches,
            bindings,
            programs,
            in_flight: None,
            last_signal: 0,
        })
    }
}
impl MetalBuffer {
    pub fn len(&self) -> usize {
        self.inner.size
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        span(self.len(), offset, bytes.len())?;
        let p = send!(
            self.inner.device.api,
            self.inner.native.id(),
            "contents",
            *mut u8
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(offset), bytes.len());
        }
        Ok(())
    }
    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, String> {
        span(self.len(), offset, length)?;
        let p = send!(
            self.inner.device.api,
            self.inner.native.id(),
            "contents",
            *mut u8
        );
        Ok(unsafe { std::slice::from_raw_parts(p.add(offset), length) }.to_vec())
    }
    pub fn device_address(&self) -> u64 {
        send!(
            self.inner.device.api,
            self.inner.native.id(),
            "gpuAddress",
            u64
        )
    }
}
impl MetalCommandBuffer {
    pub fn patch_buffer(
        &mut self,
        command: usize,
        index: usize,
        buffer: &MetalBuffer,
        offset: usize,
    ) -> Result<(), String> {
        self.wait()?;
        if command >= self.launches.len() {
            return Err("Metal command index".into());
        }
        if index >= self.bindings[command].len() {
            return Err("Metal buffer index".into());
        }
        span(buffer.len(), offset, 0)?;
        if !Rc::ptr_eq(&self.device, &buffer.inner.device) {
            return Err("Metal buffer belongs to another device".into());
        }
        let cmd =
            send!(self.device.api,self.icb.id(),"indirectComputeCommandAtIndex:",Id,command=>usize);
        send!(self.device.api,cmd,"setKernelBuffer:offset:atIndex:",(),buffer.inner.native.id()=>Id,offset=>usize,index=>usize);
        self.bindings[command][index] = buffer.clone();
        Ok(())
    }
    pub fn patch_launch(&mut self, command: usize, launch: Launch) -> Result<(), String> {
        self.wait()?;
        let p = self.programs.get(command).ok_or("Metal command index")?;
        launch.validate(p.inner.max_threads)?;
        let cmd =
            send!(self.device.api,self.icb.id(),"indirectComputeCommandAtIndex:",Id,command=>usize);
        send!(self.device.api,cmd,"concurrentDispatchThreadgroups:threadsPerThreadgroup:",(),Size::from(launch.groups)=>Size,Size::from(launch.local)=>Size);
        self.launches[command] = launch;
        Ok(())
    }
    pub fn submit(&mut self, wait_value: u64, signal: u64) -> Result<(), String> {
        self.wait()?;
        if signal <= self.device.submitted.get() {
            return Err("Metal timeline must increase".into());
        }
        let i = &self.device;
        if wait_value > i.submitted.get() {
            return Err("Metal wait references an unsubmitted timeline value".into());
        }
        let cb = send!(i.api, i.queue.id(), "commandBuffer", Id);
        send!(i.api, cb, "retain", Id);
        let cb = Owned::new(&i.api, cb, "command buffer")?;
        if wait_value > 0 {
            send!(i.api,cb.id(),"encodeWaitForEvent:value:",(),i.event.id()=>Id,wait_value=>u64);
        }
        let enc = send!(i.api, cb.id(), "computeCommandEncoder", Id);
        for b in self.bindings.iter().flatten() {
            send!(i.api,enc,"useResource:usage:",(),b.inner.native.id()=>Id,3=>usize);
        }
        for p in &self.programs {
            send!(i.api,enc,"setComputePipelineState:",(),p.inner.native.id()=>Id);
        }
        send!(i.api,enc,"executeCommandsInBuffer:withRange:",(),self.icb.id()=>Id,Range{location:0,length:self.launches.len()}=>Range);
        send!(i.api, enc, "endEncoding", ());
        send!(i.api,cb.id(),"encodeSignalEvent:value:",(),i.event.id()=>Id,signal=>u64);
        send!(i.api, cb.id(), "commit", ());
        self.in_flight = Some(cb);
        self.last_signal = signal;
        self.device.submitted.set(signal);
        Ok(())
    }
    pub fn wait(&mut self) -> Result<(), String> {
        if let Some(cb) = self.in_flight.take() {
            send!(self.device.api, cb.id(), "waitUntilCompleted", ());
            if send!(self.device.api, cb.id(), "status", usize) != 4 {
                return Err("Metal command execution failed".into());
            }
        }
        Ok(())
    }
    pub fn completed_value(&self) -> u64 {
        send!(
            self.device.api,
            self.device.event.id(),
            "signaledValue",
            u64
        )
    }
    pub fn wait_signal(&self, value: u64, timeout: Duration) -> Result<(), String> {
        let start = Instant::now();
        while self.completed_value() < value {
            if start.elapsed() > timeout {
                return Err("Metal shared event timeout".into());
            }
            std::thread::yield_now();
        }
        Ok(())
    }
}
impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}
```

`tests/drivers.rs`を保存する。

```rust
// file="tests/drivers.rs" mode="write"
use chibigrad::gpu::{Kernel, Launch};
static GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn launch() -> Launch {
    Launch {
        groups: [2, 1, 1],
        local: [4, 1, 1],
    }
}
fn floats(bytes: Vec<u8>) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
fn words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_indirect_replay_patch_and_shared_event() {
    let _gpu = GPU_TEST.lock().unwrap();
    use chibigrad::backends::metal::MetalDevice;
    let device = MetalDevice::open().expect("Metal device must be available on this test host");
    eprintln!("Metal hardware: {:?}", device.capabilities());
    let spec=Kernel{source:"#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device const float *a [[buffer(0)]],device float *o [[buffer(1)]],device const float *s [[buffer(2)]],uint gid [[thread_position_in_grid]]){o[gid]=a[gid]+s[0];}".into(),entry:"add".into(),launch:launch(),shared_bytes:0};
    let program = device.compile(&spec).unwrap();
    let input = device.allocate(32).unwrap();
    let output = device.allocate(32).unwrap();
    let scalar = device.allocate(4).unwrap();
    input
        .write(0, &words(&[1., 2., 3., 4., 5., 6., 7., 8.]))
        .unwrap();
    scalar.write(0, &words(&[2.])).unwrap();
    let mut commands = device
        .command_buffer(&[(
            program,
            launch(),
            vec![input.clone(), output.clone(), scalar.clone()],
        )])
        .unwrap();
    commands.submit(0, 1).unwrap();
    commands.wait().unwrap();
    commands
        .wait_signal(1, std::time::Duration::from_secs(2))
        .unwrap();
    assert_eq!(
        floats(output.read(0, 32).unwrap()),
        vec![3., 4., 5., 6., 7., 8., 9., 10.]
    );
    let replacement = device.allocate(32).unwrap();
    replacement.write(0, &words(&[10.; 8])).unwrap();
    commands.patch_buffer(0, 0, &replacement, 0).unwrap();
    scalar.write(0, &words(&[5.])).unwrap();
    drop(replacement);
    drop(input);
    drop(device);
    commands.submit(1, 2).unwrap();
    commands.wait().unwrap();
    assert!(commands.completed_value() >= 2);
    assert_eq!(floats(output.read(0, 32).unwrap()), vec![15.; 8]);
    assert!(commands.submit(2, 2).is_err());
}
```

```sh
# check step=15.02 requires=metal
cargo test --features metal --test drivers -- --nocapture
```

最初の結果は 3〜10、差し替え後は全要素 15 になる。

## 15.03 共通 UOp から Metal ソースへ

既存の LOAD、STORE、RANGE を Metal へ描画する。出力軸を Global に変え、独立した要素を GPU スレッドへ割り当てる。Global の UIR 往復も同時に追加する。

`src/lib.rs`を追記する。

```rust
// file="src/lib.rs" mode="append"


pub mod renderers;
```

`src/renderers.rs`を保存する。

```rust
// file="src/renderers.rs" mode="write"
use crate::gpu::{Kernel, Launch};
use crate::{Arg, Axis, DType, Ops, UOp};
use std::collections::{BTreeMap, HashMap, HashSet};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Metal,
}
fn ident(name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || name.as_bytes()[0].is_ascii_digit()
    {
        Err("invalid GPU identifier".into())
    } else {
        Ok(())
    }
}
fn ty(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::I64 => "long",
        DType::Bool => "bool",
        DType::Ptr => "float*",
        DType::Void => "void",
        _ => "int",
    }
}
/// Render already lowered common UOps. Tensor shapes and view operations never enter this renderer.
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
    };
    source.push_str("inline long chibi_div(long a,long b){long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\ninline long chibi_mod(long a,long b){long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\n");
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
        })
        .collect::<Vec<_>>();
    {
        signature.extend([
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
                .ok_or_else(|| format!("GPU UOp {:?} source {} not linearized", u.op, s.id))
        };
        let expr = match u.op {
            Ops::Const => match u.arg {
                Arg::F32(bits) => match dialect {
                    Dialect::Metal => format!("as_type<float>((uint){bits})"),
                },
                Arg::I32(x) => format!("{x}"),
                Arg::Int(x) => format!("{x}L"),
                Arg::Bool(b) => if b { "true" } else { "false" }.into(),
                _ => return Err("unsupported GPU constant".into()),
            },
            Ops::Alloc => {
                if let Arg::Allocation { dtype, size, .. } = u.arg {
                    if size != 1 {
                        return Err("scalar GPU allocations must contain one element".into());
                    }
                    let address = "";
                    declarations.push_str(&format!("{address}{} {name}[{size}];\n", ty(dtype)));
                    values.insert(u.id, name);
                    continue;
                } else {
                    return Err("GPU allocation missing layout".into());
                }
            }
            Ops::Param | Ops::Buffer => format!(
                "p{}",
                buffer_slots.get(&u.id).ok_or("GPU buffer missing slot")?
            ),
            Ops::Index => format!("({}+({}))", get(0)?, get(1)?),
            Ops::Load => {
                let index = &u.src[0];
                let ptr = get(0)?;
                let cast = match u.arg {
                    Arg::DType(dt) => format!("{}*", ty(dt)),
                    _ => "float*".into(),
                };
                let addr = "device ";
                let typed = if matches!(u.arg, Arg::DType(_)) {
                    format!("(({addr}{cast}){ptr})")
                } else {
                    ptr
                };
                if index.op == Ops::Index && index.src.len() > 2 && index.src[2] != UOp::bool(true)
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
                let index = &u.src[0];
                if index.op == Ops::Index && index.src.len() > 2 && index.src[2] != UOp::bool(true)
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
                        };
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!("{name}={hw};\n"));
                    }
                    _ => {
                        declarations.push_str(&format!("long {name};\n"));
                        body.push_str(&format!(
                            "for({name}={};{name}<{};{name}++){{\n",
                            get(0)?,
                            get(1)?
                        ));
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
            Ops::Div if matches!(u.dtype(), DType::I32 | DType::I64) => {
                format!("chibi_div({},{})", get(0)?, get(1)?)
            }
            Ops::Mod => format!("chibi_mod({},{})", get(0)?, get(1)?),
            Ops::Add | Ops::Sub | Ops::Mul | Ops::Div | Ops::Lt | Ops::Eq | Ops::And | Ops::Or => {
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
                format!("({} {op} {})", get(0)?, get(1)?)
            }
            Ops::Max => format!("max({},{})", get(0)?, get(1)?),
            Ops::Neg => format!("(-{})", get(0)?),
            Ops::Exp | Ops::Log | Ops::Sin | Ops::Cos | Ops::Sqrt | Ops::Exp2 | Ops::Log2 => {
                let fun = match u.op {
                    Ops::Exp => "exp",
                    Ops::Log => "log",
                    Ops::Sin => "sin",
                    Ops::Cos => "cos",
                    Ops::Sqrt => "sqrt",
                    Ops::Exp2 => "exp2",
                    _ => "log2",
                };
                format!("{fun}({})", get(0)?)
            }
            Ops::Where => format!("({}?{}:{})", get(0)?, get(1)?, get(2)?),
            Ops::Cast => format!("(({}){})", ty(u.dtype()), get(0)?),
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
    source.push_str(&format!(
        "{} void {entry}({}){{\n{declarations}{body}}}\n",
        match dialect {
            Dialect::Metal => "kernel",
        },
        signature.join(",")
    ));
    Ok(Kernel {
        source,
        entry: entry.into(),
        launch,
        shared_bytes,
    })
}
```

`src/uir.rs` の `fn decode_arg`を置き換える。

```rust
// file="src/uir.rs" mode="replace" item="fn decode_arg"
fn decode_arg(s: &str) -> Result<Arg> {
    let fields: Vec<_> = s.split(';').collect();
    Ok(match fields.as_slice() {
        ["none"] => Arg::None,
        ["f32", v] if v.len() == 8 => {
            Arg::F32(u32::from_str_radix(v, 16).map_err(|_| "invalid f32 bits")?)
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
        ["scalar", i, dt] => Arg::ScalarParam {
            index: number(i)?,
            dtype: dtype(dt)?,
        },
        _ => return Err("invalid typed argument".into()),
    })
}
```

`src/uop.rs` の `pub enum Axis`を置き換える。

```rust
// file="src/uop.rs" mode="replace" item="pub enum Axis"
pub enum Axis {
    Loop,
    Reduce,
    Global,
}
```

`tests/gpu_common.rs`を保存する。

```rust
// file="tests/gpu_common.rs" mode="write"
#![cfg(all(feature = "metal", target_os = "macos"))]
use chibigrad::{Arg, Ops, UOp};
use chibigrad::{
    backends::metal::MetalDevice,
    gpu::Launch,
    renderers::{Dialect, render},
};
#[test]
fn lowered_uop_is_rendered_and_executed_on_metal() {
    let param = |index| {
        UOp::new(
            Ops::Param,
            vec![],
            Arg::Slot {
                index,
                shape: vec![4],
            },
        )
    };
    let a = param(0);
    let out = param(1);
    let r = UOp::range(UOp::int(4), 0, chibigrad::Axis::Global);
    let store = out.index(r.clone(), UOp::bool(true)).store(
        a.index(r.clone(), UOp::bool(true))
            .load()
            .mul(&UOp::f32(3.)),
    );
    let end = UOp::new(Ops::End, vec![store.clone(), r.clone()], Arg::None);
    let root = UOp::linear(vec![r, store, end]);
    let mut ops = vec![];
    let mut seen = std::collections::HashSet::new();
    for node in &root.src {
        for u in node.toposort() {
            if seen.insert(u.id) {
                ops.push(u)
            }
        }
    }
    let linear = UOp::linear(ops);
    assert_eq!(
        chibigrad::uir::print(&chibigrad::uir::parse(&chibigrad::uir::print(&linear)).unwrap()),
        chibigrad::uir::print(&linear)
    );
    let launch = Launch {
        groups: [4, 1, 1],
        local: [1, 1, 1],
    };
    let kernel = render(&linear, "triple", Dialect::Metal, launch.clone()).unwrap();
    let device = MetalDevice::open().unwrap();
    let program = device.compile(&kernel).unwrap();
    let a = device.allocate(16).unwrap();
    let out = device.allocate(16).unwrap();
    a.write(
        0,
        &[1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut command = device
        .command_buffer(&[(program, launch, vec![a, out.clone()])])
        .unwrap();
    command.submit(0, 1).unwrap();
    command.wait().unwrap();
    let values = out
        .read(0, 16)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![3., 6., 9., 12.]);
}
```

```sh
# check step=15.03 requires=metal
cargo test --features metal --test gpu_common -- --nocapture
```

共通 UOp から生成したカーネルが [3,6,9,12] を返す。

## 15.04 Tensor を Rust ホストから GPU 実行

既存スケジュールの寿命計画を GPU バッファへ対応付ける。転送、ネイティブ送信、読み戻しを通常の Rust 呼び出しで実行し、CPU の結果と比較する。

`src/gpu_tensor.rs`を保存する。

```rust
// file="src/gpu_tensor.rs" mode="write"
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::backends::metal::MetalDevice;
use crate::{
    Arg, Axis, Buffer, Ops, Result, Tensor, UOp,
    codegen::{rangeify, signature},
    runtime::InputSpec,
    schedule::{Schedule, create_schedule},
    uop::numel,
};
use crate::{
    gpu::{Kernel, Launch},
    renderers::{Dialect, render},
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
}
enum Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalDevice),
}
#[derive(Clone)]
enum Allocation {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalBuffer),
}
impl Device {
    fn allocate(&self, bytes: usize) -> Result<Allocation> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(d) => d.allocate(bytes.max(4)).map(Allocation::Metal),
        }
    }
}
impl Allocation {
    fn write(&self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => b.write(0, bytes),
        }
    }
    fn read(&self, bytes: usize) -> Result<Vec<u8>> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => b.read(0, bytes),
        }
    }
}
enum Commands {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalCommandBuffer),
}
impl Commands {
    fn run(&mut self, previous: i64, next: i64) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(command) => {
                command.submit(previous as u64, next as u64)?;
                command.wait()
            }
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
    commands: Commands,
    input_nodes: Vec<UOp>,
    captured: Vec<Buffer>,
    transfers: Vec<(Allocation, HostBuffer, bool)>,
    sequence: i64,
}

fn transfer(
    transfers: &mut Vec<(Allocation, HostBuffer, bool)>,
    buffer: &Allocation,
    host: HostBuffer,
    read: bool,
) {
    transfers.push((buffer.clone(), host, read));
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
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
        };
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
            transfer(&mut transfers, &allocation, host, false);
            allocations.insert(node.id, allocation);
        }
        let error = device.allocate(4)?;
        transfer(&mut transfers, &error, HostBuffer::Error, false);
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
        let commands = match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(device) => {
                let native = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let buffers = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::Metal(b) => b.clone(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Commands::Metal(device.command_buffer(&native)?)
            }
        };
        for (i, output) in schedule.outputs.iter().enumerate() {
            transfer(
                &mut transfers,
                &allocations[&output.id],
                HostBuffer::Output(i),
                true,
            );
        }
        transfer(&mut transfers, &error, HostBuffer::Error, true);
        Ok(Self {
            inputs,
            schedule,
            programs,
            commands,
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
        for read in [false, true] {
            if read {
                self.commands.run(self.sequence, next)?;
                self.sequence = next;
            }
            for (allocation, host, is_read) in &self.transfers {
                if *is_read != read {
                    continue;
                }
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
                if read {
                    let data = allocation.read(bytes)?;
                    if !data.is_empty() {
                        unsafe {
                            std::ptr::copy_nonoverlapping(data.as_ptr(), pointer as *mut u8, bytes)
                        };
                    }
                } else if bytes > 0 {
                    let data = unsafe { std::slice::from_raw_parts(pointer as *const u8, bytes) };
                    allocation.write(data)?;
                }
            }
        }
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
pub mod gpu_tensor;


#[cfg(any(all(feature="metal",target_os="macos"),feature="opencl"))]
pub use gpu_tensor::{Backend,GpuExecutable};
```

`tests/tensor_gpu.rs`を保存する。

```rust
// file="tests/tensor_gpu.rs" mode="write"
#![cfg(all(feature = "metal", target_os = "macos"))]
use chibigrad::{Backend, Buffer, GpuExecutable, Tensor};
use std::collections::HashMap;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{x} != {y}")
    }
}
fn backends() -> [Backend; 1] {
    [Backend::Metal]
}

#[test]
fn fused_reduction_rebinds_native_inputs() {
    let x = Tensor::parameter("x", &[4]);
    let w = Tensor::parameter("w", &[4]);
    let loss = x.mul(&w).add(&Tensor::scalar(1.)).sum();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&[loss.clone()], backend).unwrap();
        assert_eq!(exe.kernel_count(), 1);
        for (data, expected) in [(vec![1., 2., 3., 4.], 44.), (vec![2.; 4], 36.)] {
            let inputs = exe
                .inputs()
                .iter()
                .map(|spec| {
                    Buffer::new(
                        spec.shape.clone(),
                        if spec.name == "x" {
                            data.clone()
                        } else {
                            vec![4.; 4]
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            close(
                &exe.run(&inputs, &HashMap::new()).unwrap()[0].data,
                &[expected],
            );
        }
    }
}

#[test]
fn matmul_stage_movement_multiple_outputs_rebind() {
    let x = Tensor::parameter("x", &[2, 3]);
    let w = Tensor::from_slice(&[1., 0., 2., 0., 1., 3.], &[3, 2]);
    let matmul = x.matmul(&w).stage();
    let moved = matmul.permute(&[1, 0]).flip(&[1]);
    let outputs = [moved.clone(), moved.sum(), matmul.add(&Tensor::scalar(2.))];
    let cpu = chibigrad::compile_many(&outputs).unwrap();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&outputs, backend).unwrap();
        assert!(exe.kernel_count() > 1);
        for scale in [1., 2.] {
            let input =
                Buffer::new(vec![2, 3], (1..=6).map(|v| v as f32 * scale).collect()).unwrap();
            let expected = cpu.run(&[input.clone()], &HashMap::new()).unwrap();
            let actual = exe.run(&[input], &HashMap::new()).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert_eq!(a.shape, b.shape);
                close(&a.data, &b.data);
            }
        }
    }
}

#[test]
fn tensor_gradients_match_cpu() {
    let x = Tensor::from_slice(&[1., 2., 3., 4.], &[2, 2]);
    let w = Tensor::from_slice(&[2., 3.], &[2]);
    let loss = x.mul(&w).mul(&x).sum();
    let grad = loss.gradients(&[x, w]).unwrap();
    let outputs = [loss, grad[0].clone(), grad[1].clone()];
    let expected = chibigrad::compile_many(&outputs)
        .unwrap()
        .run(&[], &HashMap::new())
        .unwrap();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&outputs, backend).unwrap();
        let actual = exe.run(&[], &HashMap::new()).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            close(&a.data, &b.data);
        }
    }
}

#[test]
fn gather_scatter_validation() {
    let x = Tensor::from_slice(&[3., 4., 5.], &[3]);
    let indices = Tensor::parameter("indices", &[1]);
    let gather = x.gather(&indices, 0);
    let gradient = gather.sum().gradient(&x).unwrap();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&[gather.clone(), gradient.clone()], backend).unwrap();
        for bad in [-1., 3., 0.5, f32::NAN, f32::INFINITY] {
            assert!(
                exe.run(&[Buffer::new(vec![1], vec![bad]).unwrap()], &HashMap::new())
                    .is_err()
            );
        }
        let actual = exe
            .run(&[Buffer::new(vec![1], vec![2.]).unwrap()], &HashMap::new())
            .unwrap();
        close(&actual[0].data, &[5.]);
        close(&actual[1].data, &[0., 0., 1.]);
    }
}

#[test]
fn reused_slots_parallel_row_reductions_and_convolution_gradients() {
    let x = Tensor::from_slice(&(1..=12).map(|v| v as f32).collect::<Vec<_>>(), &[3, 4]);
    let a = x.add(&Tensor::scalar(1.)).stage();
    let b = a.mul(&Tensor::scalar(2.)).stage();
    let c = b.sub(&Tensor::scalar(4.)).stage();
    let d = c.add(&Tensor::scalar(8.)).stage();
    let image = Tensor::from_slice(
        &(1..=9).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 3, 3],
    );
    let weight = Tensor::from_slice(&[1., 0., 0., -1.], &[1, 1, 2, 2]);
    let conv = image.conv2d(&weight, (1, 1), (0, 0));
    let gradient = conv.sum().gradient(&weight).unwrap();
    let outputs = [d.sum_axes(&[1], false), conv, gradient];
    let expected = chibigrad::compile_many(&outputs)
        .unwrap()
        .run(&[], &HashMap::new())
        .unwrap();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&outputs, backend).unwrap();
        let slots = exe
            .schedule()
            .lifetimes
            .iter()
            .map(|l| l.slot)
            .collect::<std::collections::HashSet<_>>();
        assert!(slots.len() < exe.schedule().lifetimes.len());
        assert!(
            exe.programs()
                .iter()
                .any(|p| p.src[1].src.iter().any(|u| matches!(
                    u.arg,
                    chibigrad::Arg::Range {
                        axis: chibigrad::Axis::Global,
                        ..
                    }
                )))
        );
        let actual = exe.run(&[], &HashMap::new()).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            close(&a.data, &b.data);
        }
    }
}

#[test]
fn unary_broadcast_stack_and_empty_outputs() {
    let x = Tensor::from_slice(&[0.25, 0.5, 1., 2.], &[2, 2]);
    let funcs = x
        .exp()
        .log()
        .sin()
        .cos()
        .add(&x.sqrt())
        .maximum(&Tensor::scalar(0.));
    let moved = x
        .pad(&[(1, 0), (1, 1)])
        .flip(&[0])
        .shrink(&[(0, 2), (1, 3)]);
    let stacked = Tensor::stack(&[funcs, moved], 0);
    let outputs = [stacked, Tensor::from_slice(&[], &[0])];
    let expected = chibigrad::compile_many(&outputs)
        .unwrap()
        .run(&[], &HashMap::new())
        .unwrap();
    for backend in backends() {
        let mut exe = GpuExecutable::compile(&outputs, backend).unwrap();
        let actual = exe.run(&[], &HashMap::new()).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.shape, b.shape);
            close(&a.data, &b.data);
        }
    }
}
```

```sh
# check step=15.04 requires=metal
cargo test --features metal --test tensor_gpu -- --nocapture
```

入力の再束縛、複数出力、移動、畳み込み、勾配、空出力、添字検査の 6 テストが通る。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 15-001 利用可能な Metal device を検出し、未対応時の明示的エラーを作る。

<!-- todo:15-001 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-002 静的 f32 の加算用 MSL を共通 kernel IR から生成する。

<!-- todo:15-002 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-003 公開 API で library と pipeline を作り、build error を返す。

<!-- todo:15-003 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-004 shared buffer の入力を書き、4 thread を dispatch する。

<!-- todo:15-004 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-005 完了待ち後に出力を読み、CPU の値と比較する。

<!-- todo:15-005 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-006 groups を `ceil(N/L)` で計算し、掛算 overflow を検査する。

<!-- todo:15-006 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-007 `i<N` の内側に LOAD と STORE の両方を置く。

<!-- todo:15-007 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-008 device と pipeline の上限を使って local size を検証する。

<!-- todo:15-008 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-009 N=1、L-1、L、L+1、257 を同じ lowering で実行する。

<!-- todo:15-009 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-010 N=0 は host 側で空出力を返し、ゼロ個の group を dispatch しない。

<!-- todo:15-010 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-011 Plan の入力 buffer を差し替えて再実行する。

<!-- todo:15-011 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-012 出力の平坦 ID を多次元座標へ復元する。

<!-- todo:15-012 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-013 movement と broadcast を CPU と同じ座標変換から生成する。

<!-- todo:15-013 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-014 pad の有効条件が成立した場合だけ device buffer を LOAD する。

<!-- todo:15-014 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-015 無効 LOAD の値を事後に WHERE で隠す実装を拒否する。

<!-- todo:15-015 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-016 CPU、interpreter、Metal の三つの値を比較する。

<!-- todo:15-016 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-017 一つの thread が一行を担当し、列方向を逐次累積する。

<!-- todo:15-017 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-018 matmul は一つの thread が一出力 `(m,n)` を担当する。

<!-- todo:15-018 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-019 reduction の初期値を各 thread のローカル変数に置く。

<!-- todo:15-019 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-020 CPU と同じ shape graph から reduction と matmul を作る。

<!-- todo:15-020 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-021 小型 MLP の前向きと勾配にも同じ backend を選べるようにする。

<!-- todo:15-021 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-022 CPU 用テストは Metal がない環境でも実行できる。

<!-- todo:15-022 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-023 未対応機器の skip を、実機の数値比較成功として報告しない。

<!-- todo:15-023 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-024 GPU の build error と command buffer の実行エラーを別々に伝える。

<!-- todo:15-024 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-025 結果を読む前の完了待ちとリソースの寿命が runtime に集約される。

<!-- todo:15-025 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-026 MSL に未対応 UOp が残る場合はコンパイル前に拒否する。

<!-- todo:15-026 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 15-027 空の加算をテストし、Metal 呼出しなしで空の結果を得る。

<!-- todo:15-027 -->

関連する実装と確認は本章の `15.01`, `15.02`, `15.03`, `15.04` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
