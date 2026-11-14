# 16. OpenCLを追加し、同じTensorをGPU実行する

前の節で動かしたコードに、今回使う演算と処理を追加します。各節の確認を実行してから次へ進みます。

参照するUOpの構成は [tinygrad c22edff](https://github.com/tinygrad/tinygrad/blob/c22edff14c79310a9b2df62a78819cf77c8289fa/tinygrad/uop/ops.py) です。最初から全機能を実装せず、使う節で増やします。

## 16.01 OpenCL のネイティブ実行

OpenCL のプラットフォーム列挙、バッファ、コンパイル、引数束縛、enqueue、finish を追加する。Metal と同じ寿命と再生の性質を検証する。

`Cargo.toml`を追記する。

```toml
# file="Cargo.toml" mode="append"
opencl=["native"]
```

`src/backends/mod.rs`を追記する。

```rust
// file="src/backends/mod.rs" mode="append"
#[cfg(feature = "opencl")]
pub mod opencl;
```

`src/backends/opencl.rs`を保存する。

```rust
// file="src/backends/opencl.rs" mode="write"
use super::dynamic::Library;
use crate::gpu::{Capabilities, Kernel, Launch, span};
use std::{
    cell::Cell,
    ffi::{CString, c_char, c_void},
    rc::Rc,
};
type Handle = usize;
struct Api {
    lib: Library,
    #[cfg(target_os = "macos")]
    _graphics: Library,
}
macro_rules! cl {
    ($a:expr,$name:literal,$r:ty $(,$v:expr=>$t:ty)*) => {{
        let ptr=$a.lib.symbol($name)?;
        let f:unsafe extern "C" fn($($t),*)->$r=unsafe{std::mem::transmute(ptr)};
        unsafe{f($($v),*)}
    }};
}
fn check(code: i32, name: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{name}: OpenCL error {code}"))
    }
}
impl Api {
    fn load() -> Result<Rc<Self>, String> {
        #[cfg(target_os = "macos")]
        let graphics =
            Library::open("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
        #[cfg(target_os = "macos")]
        let names = ["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
        #[cfg(not(target_os = "macos"))]
        let names = ["libOpenCL.so.1", "libOpenCL.so"];
        for name in names {
            if let Ok(lib) = Library::open(name) {
                return Ok(Rc::new(Self {
                    lib,
                    #[cfg(target_os = "macos")]
                    _graphics: graphics,
                }));
            }
        }
        Err("OpenCL runtime unavailable".into())
    }
    fn info(&self, device: Handle, key: u32) -> Result<Vec<u8>, String> {
        let mut n = 0usize;
        check(
            cl!(self,"clGetDeviceInfo",i32,device=>Handle,key=>u32,0=>usize,std::ptr::null_mut::<c_void>()=>*mut c_void,&mut n=>*mut usize),
            "clGetDeviceInfo",
        )?;
        let mut bytes = vec![0u8; n];
        check(
            cl!(self,"clGetDeviceInfo",i32,device=>Handle,key=>u32,n=>usize,bytes.as_mut_ptr().cast()=>*mut c_void,std::ptr::null_mut::<usize>()=>*mut usize),
            "clGetDeviceInfo",
        )?;
        Ok(bytes)
    }
    fn text(&self, device: Handle, key: u32) -> Result<String, String> {
        let mut v = self.info(device, key)?;
        while v.last() == Some(&0) {
            v.pop();
        }
        Ok(String::from_utf8_lossy(&v).into_owned())
    }
    fn number(&self, device: Handle, key: u32) -> Result<usize, String> {
        let bytes = self.info(device, key)?;
        let mut v = [0u8; 8];
        let n = bytes.len().min(8);
        v[..n].copy_from_slice(&bytes[..n]);
        Ok(u64::from_ne_bytes(v) as usize)
    }
    fn devices(&self) -> Result<Vec<Handle>, String> {
        let mut n = 0u32;
        check(
            cl!(self,"clGetPlatformIDs",i32,0=>u32,std::ptr::null_mut::<Handle>()=>*mut Handle,&mut n=>*mut u32),
            "clGetPlatformIDs",
        )?;
        let mut platforms = vec![0usize; n as usize];
        check(
            cl!(self,"clGetPlatformIDs",i32,n=>u32,platforms.as_mut_ptr()=>*mut Handle,std::ptr::null_mut::<u32>()=>*mut u32),
            "clGetPlatformIDs",
        )?;
        let mut out = vec![];
        for platform in platforms {
            let mut count = 0u32;
            let status = cl!(self,"clGetDeviceIDs",i32,platform=>Handle,0xffffffffu64=>u64,0=>u32,std::ptr::null_mut::<Handle>()=>*mut Handle,&mut count=>*mut u32);
            if status == -1 {
                continue;
            }
            check(status, "clGetDeviceIDs")?;
            if count == 0 {
                continue;
            }
            let mut devices = vec![0usize; count as usize];
            check(
                cl!(self,"clGetDeviceIDs",i32,platform=>Handle,0xffffffffu64=>u64,count=>u32,devices.as_mut_ptr()=>*mut Handle,std::ptr::null_mut::<u32>()=>*mut u32),
                "clGetDeviceIDs",
            )?;
            out.extend(devices);
        }
        Ok(out)
    }
}
#[derive(Clone)]
pub struct OpenClDevice {
    inner: Rc<DeviceInner>,
}
struct DeviceInner {
    api: Rc<Api>,
    device: Handle,
    context: Handle,
    queue: Handle,
    caps: Capabilities,
    enqueued: Cell<u64>,
    completed: Cell<u64>,
}
impl Drop for DeviceInner {
    fn drop(&mut self) {
        let _ = (|| -> Result<(), String> {
            cl!(self.api,"clFinish",i32,self.queue=>Handle);
            cl!(self.api,"clReleaseCommandQueue",i32,self.queue=>Handle);
            cl!(self.api,"clReleaseContext",i32,self.context=>Handle);
            Ok(())
        })();
    }
}
#[derive(Clone)]
pub struct OpenClBuffer {
    inner: Rc<BufferInner>,
}
struct BufferInner {
    native: Handle,
    size: usize,
    device: Rc<DeviceInner>,
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        let _ = (|| -> Result<(), String> {
            cl!(self.device.api,"clReleaseMemObject",i32,self.native=>Handle);
            Ok(())
        })();
    }
}
#[derive(Clone)]
pub struct OpenClProgram {
    inner: Rc<ProgramInner>,
}
struct ProgramInner {
    native: Handle,
    program: Handle,
    device: Rc<DeviceInner>,
}
impl Drop for ProgramInner {
    fn drop(&mut self) {
        let _ = (|| -> Result<(), String> {
            cl!(self.device.api,"clReleaseKernel",i32,self.native=>Handle);
            cl!(self.device.api,"clReleaseProgram",i32,self.program=>Handle);
            Ok(())
        })();
    }
}
#[derive(Clone)]
pub enum Argument {
    Buffer(OpenClBuffer),
    Scalar(Vec<u8>),
}
pub struct OpenClCommand {
    program: OpenClProgram,
    launch: Launch,
    args: Vec<Argument>,
    signal: u64,
}
impl OpenClDevice {
    pub fn devices() -> Result<Vec<String>, String> {
        let _initializing = super::INITIALIZE
            .lock()
            .map_err(|_| "device initialization poisoned")?;
        let api = Api::load()?;
        api.devices()?
            .into_iter()
            .map(|d| api.text(d, 0x102b))
            .collect()
    }
    pub fn open(index: usize) -> Result<Self, String> {
        let _initializing = super::INITIALIZE
            .lock()
            .map_err(|_| "device initialization poisoned")?;
        let api = Api::load()?;
        let device = *api
            .devices()?
            .get(index)
            .ok_or("OpenCL device unavailable")?;
        let mut err = 0i32;
        let context = cl!(api,"clCreateContext",Handle,std::ptr::null::<isize>()=>*const isize,1=>u32,&device=>*const Handle,std::ptr::null::<c_void>()=>*const c_void,std::ptr::null_mut::<c_void>()=>*mut c_void,&mut err=>*mut i32);
        check(err, "clCreateContext")?;
        let queue = cl!(api,"clCreateCommandQueue",Handle,context=>Handle,device=>Handle,0u64=>u64,&mut err=>*mut i32);
        check(err, "clCreateCommandQueue")?;
        let extensions = api
            .text(device, 0x1030)?
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let caps = Capabilities {
            name: api.text(device, 0x102b)?,
            max_threads: api.number(device, 0x1004)?,
            local_memory: api.number(device, 0x1023)?,
            extensions,
            ..Capabilities::default()
        };
        Ok(Self {
            inner: Rc::new(DeviceInner {
                api,
                device,
                context,
                queue,
                caps,
                enqueued: Cell::new(0),
                completed: Cell::new(0),
            }),
        })
    }
    pub fn capabilities(&self) -> &Capabilities {
        &self.inner.caps
    }
    pub fn allocate(&self, size: usize) -> Result<OpenClBuffer, String> {
        let i = &self.inner;
        let mut err = 0i32;
        let native = cl!(i.api,"clCreateBuffer",Handle,i.context=>Handle,1u64=>u64,size.max(1)=>usize,std::ptr::null_mut::<c_void>()=>*mut c_void,&mut err=>*mut i32);
        check(err, "clCreateBuffer")?;
        Ok(OpenClBuffer {
            inner: Rc::new(BufferInner {
                native,
                size,
                device: i.clone(),
            }),
        })
    }
    pub fn compile(&self, spec: &Kernel) -> Result<OpenClProgram, String> {
        let i = &self.inner;
        spec.launch.validate(i.caps.max_threads)?;
        if spec.shared_bytes > i.caps.local_memory {
            return Err("OpenCL local memory limit".into());
        }
        let source = CString::new(spec.source.as_str()).map_err(|_| "NUL in source")?;
        let p = source.as_ptr();
        let mut err = 0i32;
        let program = cl!(i.api,"clCreateProgramWithSource",Handle,i.context=>Handle,1=>u32,&p=>*const *const c_char,std::ptr::null::<usize>()=>*const usize,&mut err=>*mut i32);
        check(err, "clCreateProgramWithSource")?;
        let status = cl!(i.api,"clBuildProgram",i32,program=>Handle,1=>u32,&i.device=>*const Handle,c"".as_ptr()=>*const c_char,std::ptr::null::<c_void>()=>*const c_void,std::ptr::null_mut::<c_void>()=>*mut c_void);
        if status != 0 {
            let mut n = 0usize;
            cl!(i.api,"clGetProgramBuildInfo",i32,program=>Handle,i.device=>Handle,0x1183u32=>u32,0=>usize,std::ptr::null_mut::<u8>()=>*mut u8,&mut n=>*mut usize);
            let mut log = vec![0u8; n];
            cl!(i.api,"clGetProgramBuildInfo",i32,program=>Handle,i.device=>Handle,0x1183u32=>u32,n=>usize,log.as_mut_ptr()=>*mut u8,std::ptr::null_mut::<usize>()=>*mut usize);
            cl!(i.api,"clReleaseProgram",i32,program=>Handle);
            return Err(format!(
                "OpenCL build failed ({status}): {}",
                String::from_utf8_lossy(&log)
            ));
        }
        let entry = CString::new(spec.entry.as_str()).map_err(|_| "invalid entry")?;
        let native = cl!(i.api,"clCreateKernel",Handle,program=>Handle,entry.as_ptr()=>*const c_char,&mut err=>*mut i32);
        check(err, "clCreateKernel")?;
        Ok(OpenClProgram {
            inner: Rc::new(ProgramInner {
                native,
                program,
                device: i.clone(),
            }),
        })
    }
    pub fn finish(&self) -> Result<(), String> {
        let i = &self.inner;
        check(cl!(i.api,"clFinish",i32,i.queue=>Handle), "clFinish")?;
        i.completed.set(i.enqueued.get());
        Ok(())
    }
    pub fn completed_value(&self) -> u64 {
        self.inner.completed.get()
    }
}
impl OpenClBuffer {
    pub fn len(&self) -> usize {
        self.inner.size
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        span(self.len(), offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let i = &self.inner;
        check(
            cl!(i.device.api,"clEnqueueWriteBuffer",i32,i.device.queue=>Handle,i.native=>Handle,1=>u32,offset=>usize,bytes.len()=>usize,bytes.as_ptr()=>*const u8,0=>u32,std::ptr::null::<Handle>()=>*const Handle,std::ptr::null_mut::<Handle>()=>*mut Handle),
            "clEnqueueWriteBuffer",
        )
    }
    pub fn read(&self, offset: usize, length: usize) -> Result<Vec<u8>, String> {
        span(self.len(), offset, length)?;
        let mut out = vec![0u8; length];
        if length == 0 {
            return Ok(out);
        }
        let i = &self.inner;
        check(
            cl!(i.device.api,"clEnqueueReadBuffer",i32,i.device.queue=>Handle,i.native=>Handle,1=>u32,offset=>usize,length=>usize,out.as_mut_ptr()=>*mut u8,0=>u32,std::ptr::null::<Handle>()=>*const Handle,std::ptr::null_mut::<Handle>()=>*mut Handle),
            "clEnqueueReadBuffer",
        )?;
        Ok(out)
    }
}
impl OpenClProgram {
    pub fn command(&self, launch: Launch, args: Vec<Argument>) -> Result<OpenClCommand, String> {
        launch.validate(self.inner.device.caps.max_threads)?;
        for arg in &args {
            if let Argument::Buffer(b) = arg {
                if !Rc::ptr_eq(&self.inner.device, &b.inner.device) {
                    return Err("OpenCL buffer belongs to another context".into());
                }
            }
        }
        Ok(OpenClCommand {
            program: self.clone(),
            launch,
            args,
            signal: 0,
        })
    }
}
impl OpenClCommand {
    pub fn patch(&mut self, index: usize, arg: Argument) -> Result<(), String> {
        self.finish()?;
        let slot = self.args.get_mut(index).ok_or("OpenCL argument index")?;
        if let Argument::Buffer(b) = &arg {
            if !Rc::ptr_eq(&self.program.inner.device, &b.inner.device) {
                return Err("OpenCL buffer belongs to another context".into());
            }
        }
        *slot = arg;
        Ok(())
    }
    pub fn patch_launch(&mut self, launch: Launch) -> Result<(), String> {
        self.finish()?;
        launch.validate(self.program.inner.device.caps.max_threads)?;
        self.launch = launch;
        Ok(())
    }
    pub fn enqueue(&mut self) -> Result<(), String> {
        let p = &self.program.inner;
        let i = &p.device;
        for (idx, arg) in self.args.iter().enumerate() {
            let (size, ptr) = match arg {
                Argument::Buffer(b) => (
                    std::mem::size_of::<Handle>(),
                    std::ptr::addr_of!(b.inner.native).cast::<c_void>(),
                ),
                Argument::Scalar(bytes) => (bytes.len(), bytes.as_ptr().cast()),
            };
            check(
                cl!(i.api,"clSetKernelArg",i32,p.native=>Handle,idx as u32=>u32,size=>usize,ptr=>*const c_void),
                "clSetKernelArg",
            )?;
        }
        let global = [
            self.launch.groups[0] * self.launch.local[0],
            self.launch.groups[1] * self.launch.local[1],
            self.launch.groups[2] * self.launch.local[2],
        ];
        check(
            cl!(i.api,"clEnqueueNDRangeKernel",i32,i.queue=>Handle,p.native=>Handle,3=>u32,std::ptr::null::<usize>()=>*const usize,global.as_ptr()=>*const usize,self.launch.local.as_ptr()=>*const usize,0=>u32,std::ptr::null::<Handle>()=>*const Handle,std::ptr::null_mut::<Handle>()=>*mut Handle),
            "clEnqueueNDRangeKernel",
        )?;
        self.signal = i
            .enqueued
            .get()
            .checked_add(1)
            .ok_or("OpenCL timeline overflow")?;
        i.enqueued.set(self.signal);
        Ok(())
    }
    pub fn finish(&self) -> Result<(), String> {
        let i = &self.program.inner.device;
        check(cl!(i.api,"clFinish",i32,i.queue=>Handle), "clFinish")?;
        i.completed.set(i.enqueued.get());
        Ok(())
    }
    pub fn completed_value(&self) -> u64 {
        self.program.inner.device.completed.get()
    }
}
impl Drop for OpenClCommand {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
```

`tests/drivers.rs`を追記する。

```rust
// file="tests/drivers.rs" mode="append"
#[cfg(feature = "opencl")]
#[test]
fn opencl_inorder_finish_and_replay() {
    let _gpu = GPU_TEST.lock().unwrap();
    use chibigrad::backends::opencl::{Argument, OpenClDevice};
    let names = match OpenClDevice::devices() {
        Ok(n) if !n.is_empty() => n,
        other => {
            eprintln!("OpenCL unavailable on test host: {other:?}");
            return;
        }
    };
    eprintln!("OpenCL hardware: {names:?}");
    let device = OpenClDevice::open(0).unwrap();
    let spec=Kernel{source:"__kernel void add(__global const float *a,__global float *o,float s){size_t gid=get_global_id(0);o[gid]=a[gid]+s;}".into(),entry:"add".into(),launch:launch(),shared_bytes:0};
    let p = device.compile(&spec).unwrap();
    let input = device.allocate(32).unwrap();
    let output = device.allocate(32).unwrap();
    input.write(0, &words(&[1.; 8])).unwrap();
    let mut command = p
        .command(
            launch(),
            vec![
                Argument::Buffer(input.clone()),
                Argument::Buffer(output.clone()),
                Argument::Scalar(2f32.to_ne_bytes().to_vec()),
            ],
        )
        .unwrap();
    command.enqueue().unwrap();
    command.finish().unwrap();
    assert_eq!(floats(output.read(0, 32).unwrap()), vec![3.; 8]);
    command
        .patch(2, Argument::Scalar(7f32.to_ne_bytes().to_vec()))
        .unwrap();
    drop(input);
    drop(p);
    drop(device);
    command.enqueue().unwrap();
    command.finish().unwrap();
    assert_eq!(command.completed_value(), 2);
    assert_eq!(floats(output.read(0, 32).unwrap()), vec![8.; 8]);
}
```

```sh
# check step=16.01 requires=opencl
cargo test --features metal,opencl --test drivers -- --nocapture
```

Metal と OpenCL の実機テストが通り、OpenCL の引数更新後は全要素 8 になる。

## 16.02 同じ UOp と Tensor を OpenCL へ

共通スカラー描画器へ OpenCL 方言を加え、同じスケジュールを選択したバックエンドへ送る。ホスト制御は Rust のネイティブ呼び出しのままにする。

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
    fn write(&self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => b.write(0, bytes),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.write(0, bytes),
        }
    }
    fn read(&self, bytes: usize) -> Result<Vec<u8>> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(b) => b.read(0, bytes),
            #[cfg(feature = "opencl")]
            Self::OpenCl(b) => b.read(0, bytes),
        }
    }
}
enum Commands {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::backends::metal::MetalCommandBuffer),
    #[cfg(feature = "opencl")]
    OpenCl(Vec<crate::backends::opencl::OpenClCommand>),
}
impl Commands {
    fn run(&mut self, previous: i64, next: i64) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(command) => {
                command.submit(previous as u64, next as u64)?;
                command.wait()
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(commands) => {
                for command in commands {
                    command.enqueue()?;
                    command.finish()?;
                }
                Ok(())
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
            #[cfg(not(feature = "opencl"))]
            Backend::OpenCl => return Err("OpenCL feature is disabled".into()),
            #[cfg(feature = "opencl")]
            Backend::OpenCl => Device::OpenCl(OpenClDevice::open(0)?),
        };
        let dialect = match backend {
            Backend::Metal => Dialect::Metal,
            Backend::OpenCl => Dialect::OpenCl,
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
                                #[cfg(feature = "opencl")]
                                _ => unreachable!(),
                            })
                            .collect();
                        Ok((device.compile(kernel)?, kernel.launch.clone(), buffers))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Commands::Metal(device.command_buffer(&native)?)
            }
            #[cfg(feature = "opencl")]
            Device::OpenCl(device) => {
                let native = kernels
                    .iter()
                    .map(|(kernel, bindings)| {
                        let args = bindings
                            .iter()
                            .map(|b| match b {
                                Allocation::OpenCl(b) => {
                                    crate::backends::opencl::Argument::Buffer(b.clone())
                                }
                                #[cfg(all(feature = "metal", target_os = "macos"))]
                                _ => unreachable!(),
                            })
                            .collect();
                        device.compile(kernel)?.command(kernel.launch.clone(), args)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Commands::OpenCl(native)
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

`src/renderers.rs` の `pub enum Dialect`を置き換える。

```rust
// file="src/renderers.rs" mode="replace" item="pub enum Dialect"
pub enum Dialect {
    Metal,
    OpenCl,
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
    source.push_str("inline long chibi_div(long a,long b){long q=a/b,r=a%b;return q-((r!=0)&&((r<0)!=(b<0)));}\ninline long chibi_mod(long a,long b){long r=a%b;return r+(((r!=0)&&((r<0)!=(b<0)))?b:0);}\n");
    let mut signature = params
        .iter()
        .map(|(idx, name)| match dialect {
            Dialect::Metal => format!("device float* {name} [[buffer({idx})]]"),
            Dialect::OpenCl => format!("__global float* {name}"),
        })
        .collect::<Vec<_>>();
    if dialect == Dialect::Metal {
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
                Arg::Float(bits) => {
                    let bits = (bits.value() as f32).to_bits();
                    match dialect {
                        Dialect::Metal => format!("as_type<float>((uint){bits})"),
                        Dialect::OpenCl => format!("as_float((uint){bits})"),
                    }
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
                            (Dialect::OpenCl, Axis::Global) => format!("get_global_id({dim})"),
                            _ => format!("get_local_id({dim})"),
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
            Dialect::OpenCl => "__kernel",
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

`tests/tensor_gpu.rs`を保存する。

```rust
// file="tests/tensor_gpu.rs" mode="write"
#![cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]
use chibigrad::{Backend, Buffer, GpuExecutable, Tensor};
use std::collections::HashMap;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{x} != {y}")
    }
}
fn backends() -> [Backend; 2] {
    [Backend::Metal, Backend::OpenCl]
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

#[test]
fn gpu_constant_encoding_keeps_special_values() {
    let values = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1.1, f32::NAN];
    let outputs = values.into_iter().map(Tensor::scalar).collect::<Vec<_>>();
    for backend in backends() {
        let mut executable = GpuExecutable::compile(&outputs, backend).unwrap();
        let actual = executable.run(&[], &HashMap::new()).unwrap();
        for (buffer, expected) in actual.iter().zip(values) {
            let value = buffer.data[0];
            if expected.is_nan() { assert!(value.is_nan()); }
            else { assert_eq!(value.to_bits(), expected.to_bits()); }
        }
    }
}

```

```sh
# check step=16.02 requires=metal,opencl
cargo test --features metal,opencl --test tensor_gpu -- --nocapture
```

同じ 6 テストが Metal と OpenCL の両方で通る。

## 章末の累積確認

前の章までのテストとexampleも再ビルドします。

```sh
# check
cargo test --quiet --all-targets
```

## 要件との対応

### 16-001 platform と device を列挙し、明示指定できる入口を追加する。

<!-- todo:16-001 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-002 f32 加算の kernel source を共通 IR から生成する。

<!-- todo:16-002 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-003 program build と kernel 作成、buffer の bind を実装する。

<!-- todo:16-003 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-004 入力転送、NDRange、同期読出しを一巡させる。

<!-- todo:16-004 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-005 各 OpenCL エラーに API 名と戻り値を含める。

<!-- todo:16-005 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-006 共通 launch から global_work_size を計算する関数を作る。

<!-- todo:16-006 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-007 N=257、local=64 では global=320 を渡す。

<!-- todo:16-007 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-008 `i<N` の内側でだけ LOAD と STORE を実行する。

<!-- todo:16-008 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-009 N=1、L-1、L、L+1 を CPU と比較する。

<!-- todo:16-009 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-010 N=0 は host 側で空出力を返し、ゼロサイズ NDRange を投入しない。

<!-- todo:16-010 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-011 local size が機器の上限を越えたら起動前に拒否する。

<!-- todo:16-011 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-012 二次元の平坦 ID を shape の座標へ戻す。

<!-- todo:16-012 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-013 broadcast、転置後 reshape、shrink、pad を共通 lowering から生成する。

<!-- todo:16-013 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-014 pad の LOAD 自体を有効条件の分岐内に置く。

<!-- todo:16-014 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-015 SQRT、SIN、LOG2、EXP2 を OpenCL の通常の数学関数へ下げる。

<!-- todo:16-015 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-016 native 高速近似関数は明示的な別モードにする。

<!-- todo:16-016 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-017 一つの work item に一行の逐次 SUM を担当させる。

<!-- todo:16-017 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-018 matmul の出力 `(m,n)` を一つの work item に割り当てる。

<!-- todo:16-018 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-019 reduction の累積を work item ごとに初期化する。

<!-- todo:16-019 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-020 CPU と OpenCL の出力、勾配を小さい固定値で比較する。

<!-- todo:16-020 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-021 新しい重みを同じ Plan の入力 slot に渡して再実行する。

<!-- todo:16-021 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-022 in-order queue で前の転送と kernel の順序が守られる。

<!-- todo:16-022 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-023 非同期 read を開始しただけの値を、完了した値として返さない。

<!-- todo:16-023 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-024 device の不在、build 失敗、起動失敗を別々の診断にする。

<!-- todo:16-024 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-025 機器がないときは source 検査と未実施の実機項目を分けて記録する。

<!-- todo:16-025 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-026 runtime の API 呼出し以外に OpenCL 固有のロジックを広げない。

<!-- todo:16-026 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。

### 16-027 空の加算をテストし、NDRange の投入なしで空の結果を得る。

<!-- todo:16-027 -->

関連する実装と確認は本章の `16.01`, `16.02` にある。`tests/uir.rs` と各節の確認コマンドで実行する。対応付けだけで個別要件を検証済みとは扱わない。
