"""Pinned-fd Linux ABI bridge for Python builds lacking pidfd APIs.

No signal is sent on import. Only x86_64 with matching installed ABI headers is
accepted. A missing kernel syscall raises OSError; PID-based signaling is never
a fallback. Callers must independently verify the born process owner.
"""
import ctypes,hashlib,os,platform,re,signal,sys
from pathlib import Path
NR_OPEN=434
NR_SEND=424
HEADER_PATHS=('/usr/include/x86_64-linux-gnu/asm/unistd_64.h','/usr/include/asm-generic/unistd.h')
def require(ok,message):
    if not ok:raise ValueError(message)
def abi_gate():
    require(sys.platform=='linux' and os.name=='posix' and platform.machine()=='x86_64' and
        ctypes.sizeof(ctypes.c_int)==4 and ctypes.sizeof(ctypes.c_long)==8 and ctypes.sizeof(ctypes.c_void_p)==8,
        'pidfd bridge is restricted to verified Linux x86_64 ABI')
    for name in HEADER_PATHS:
        p=Path(name)
        if not p.is_file():continue
        data=p.read_bytes();text=data.decode('ascii')
        actual={key:int(re.search(r'^\s*#\s*define\s+__NR_'+key+r'\s+(\d+)\s*$',text,re.M).group(1))
            for key in ('pidfd_open','pidfd_send_signal') if re.search(r'^\s*#\s*define\s+__NR_'+key+r'\s+(\d+)\s*$',text,re.M)}
        require(actual==dict(pidfd_open=NR_OPEN,pidfd_send_signal=NR_SEND),'installed pidfd syscall ABI differs')
        return dict(machine='x86_64',header_path=str(p),header_sha256=hashlib.sha256(data).hexdigest(),
            nr_pidfd_open=NR_OPEN,nr_pidfd_send_signal=NR_SEND)
    raise ValueError('installed x86_64/generic pidfd ABI header is required')
def libc():
    lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long;return lib
def checked_result(result):
    if result<0:
        code=ctypes.get_errno();raise OSError(code,os.strerror(code))
    return int(result)
def backend_info():
    abi=abi_gate();lib=libc()
    return dict(abi,pidfd_open_backend=('python' if callable(getattr(os,'pidfd_open',None)) else
        'libc' if hasattr(lib,'pidfd_open') else 'syscall'),pidfd_send_signal_backend=(
        'python' if callable(getattr(signal,'pidfd_send_signal',None)) else
        'libc' if hasattr(lib,'pidfd_send_signal') else 'syscall'))
def pidfd_open(pid,flags=0):
    require(type(pid) is int and 1<pid<=2**31-1 and type(flags) is int and flags==0,'typed C-int PID and zero pidfd-open flags required')
    abi_gate();native=getattr(os,'pidfd_open',None)
    if callable(native):return checked_result(native(pid,flags))
    lib=libc();function=getattr(lib,'pidfd_open',None)
    if function is not None:
        function.argtypes=[ctypes.c_int,ctypes.c_uint];function.restype=ctypes.c_int
        return checked_result(function(ctypes.c_int(pid),ctypes.c_uint(flags)))
    return checked_result(lib.syscall(ctypes.c_long(NR_OPEN),ctypes.c_int(pid),ctypes.c_uint(flags)))
def pidfd_send_signal(fd,signum,siginfo=None,flags=0):
    require(type(fd) is int and 0<=fd<=2**31-1 and isinstance(signum,int) and not isinstance(signum,bool) and
        int(signum) in (0,int(signal.SIGTERM),int(signal.SIGKILL)) and
        siginfo is None and type(flags) is int and flags==0,'pinned fd, allowed signal, no siginfo and zero flags required')
    signum=int(signum)
    abi_gate();native=getattr(signal,'pidfd_send_signal',None)
    if callable(native):native(fd,signum,None,0);return None
    lib=libc();function=getattr(lib,'pidfd_send_signal',None)
    if function is not None:
        function.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint];function.restype=ctypes.c_int
        result=function(ctypes.c_int(fd),ctypes.c_int(signum),ctypes.c_void_p(None),ctypes.c_uint(0))
    else:result=lib.syscall(ctypes.c_long(NR_SEND),ctypes.c_int(fd),ctypes.c_int(signum),ctypes.c_void_p(None),ctypes.c_uint(0))
    require(checked_result(result)==0,'unexpected nonzero pidfd signal result');return None
