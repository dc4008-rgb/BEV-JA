"""Planned local dependency initialization before original seeded evaluation.

No model/optimizer/generator/scene call is made by this module. Imports happen
only when ROOT's actual owned evaluation process calls dispatch_original_cli.
The frozen target.offline and every original scientific function stay intact.
"""
import ast
import base64
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import types

MIB=1024**2
PIDFD_PIN=(3870,'852f892365d9a4cf0017f50dd2f485f3bddb02153533a7e088a89ed970b1fa42')
ENTRY_PINS={'evaluate_ours5_saved_patches_v1.py':(15152, 'dd16b5859991839a8878a19e418e63b5a4466095fba5f009f1a9db25c0e83409')}


def need(v,message):
    if not v:raise ValueError(message)


def plain(value):
    p=Path(value)
    need(not any(x in {'.ssh','.env','id_rsa','id_ed25519','authorized_keys','credentials','token'} for x in p.parts),'no credential namespace')
    need(p.is_absolute() and str(p)==os.path.normpath(str(p)),'literal absolute bootstrap path')
    for x in [*reversed(p.parents),p]:need(not x.is_symlink(),'ordinary bootstrap/source path')
    return p


def nine(s):
    return tuple(getattr(s,k)for k in('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns'))


def whole(path,cap=2*MIB,expected=None):
    p=plain(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=nine(os.fstat(fd));need(stat.S_ISREG(before[2])and 0<before[6]<=cap,'bounded ordinary dependency source/control')
        parts=[]
        for _ in range(2):
            os.lseek(fd,0,0);chunks=[];n=0
            while True:
                b=os.read(fd,65536)
                if not b:break
                n+=len(b);need(n<=cap,'dependency source/control cap');chunks.append(b)
            parts.append(b''.join(chunks))
        os.fsync(fd);need(parts[0]==parts[1] and len(parts[0])==before[6]
            and before==nine(os.fstat(fd))==nine(p.lstat()),'two whole/full9 dependency binding')
    finally:os.close(fd)
    d={'path':str(p),'bytes':len(parts[0]),'sha256':hashlib.sha256(parts[0]).hexdigest()}
    if expected is not None:need((d['bytes'],d['sha256'])==expected,'actual reviewed source N/H')
    return parts[0],d


def save(path,value):
    b=(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    need(len(b)<=256*1024,'bounded complete initialization metadata')
    with path.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return whole(path,256*1024)[1]


def font_source_commands():
    """Read actual installed Source without importing matplotlib or guessing argv."""
    spec=importlib.util.find_spec('matplotlib')
    need(spec is not None and spec.origin is not None,'installed matplotlib Source required')
    root=plain(Path(spec.origin).parent);body,d=whole(root/'font_manager.py')
    tree=ast.parse(body);functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name=='_get_fontconfig_fonts']
    need(len(functions)==1,'actual installed Fontconfig source function')
    allowed=[]
    for call in ast.walk(functions[0]):
        if not(isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)
            and isinstance(call.func.value,ast.Name)and call.func.value.id=='subprocess'
            and call.func.attr in('check_output','run')and call.args):continue
        try:argv=ast.literal_eval(call.args[0])
        except(ValueError,TypeError):continue
        if type(argv)is list and argv and argv[0]=='fc-list' and all(type(x)is str for x in argv):
            allowed.append({'argv':argv,'actual_Source_line':call.lineno})
    need(allowed,'no source-derived local font discovery command; do not guess/exempt unknown Popen')
    binary=shutil.which('fc-list');need(binary is not None,'actual local Fontconfig binary required')
    resolved=Path(binary).resolve(strict=True)
    need(resolved.parent in(Path('/usr/bin'),Path('/bin')),'only actual system-local font discovery executable')
    _,bd=whole(resolved,16*MIB)
    return d,allowed,bd


# Exact KeyboardInterrupt lets stdlib subprocess.run/Popen.__exit__ stop its
# own unbounded context-manager wait, then our finally performs exact cleanup.
DependencyStop=KeyboardInterrupt


def pidfd_child_waitid_backend():
    """Configure real Linux WNOWAIT; no child or capability witness is invented.

    CPython may omit waitid/P_PIDFD while libc and the kernel implement them.
    A backend is only configured here. Each actual returned Popen pidfd still
    requires a successful, non-reaping kernel wait before any child is returned
    to its dependency library. Errors never fall through to another backend.
    """
    import ctypes
    machine=os.uname().machine
    need(sys.platform=='linux'and machine in('x86_64','aarch64','arm64')
         and ctypes.sizeof(ctypes.c_void_p)==ctypes.sizeof(ctypes.c_long)==8
         and ctypes.sizeof(ctypes.c_int)==ctypes.sizeof(ctypes.c_uint)==4,
         'explicit supported Linux64 waitid ABI required')
    flags=0x00000004|0x00000001|0x01000000
    for name,value in(('P_PIDFD',3),('WEXITED',4),('WNOHANG',1),('WNOWAIT',0x01000000)):
        if hasattr(os,name):need(getattr(os,name)==value,'actual Python constant matches Linux UAPI '+name)
    metadata={'actual_platform':sys.platform,'actual_machine':machine,
        'Python_has_P_PIDFD':hasattr(os,'P_PIDFD'),'Python_has_waitid':hasattr(os,'waitid'),
        'idtype_P_PIDFD':3,'options':flags,'WNOWAIT_nonreaping':True,
        'kernel_waitid_or_child_ownership_not_yet_observed':True}
    if hasattr(os,'waitid'):
        need(callable(os.waitid),'actual Python waitid callable')
        metadata['backend']='Python_os.waitid_Linux_P_PIDFD3'
        def witness(fd):
            need(type(fd)is int and 0<=fd<2**31,'actual returned pidfd integer')
            result=os.waitid(3,fd,flags)
            return {'backend':metadata['backend'],'actual_pidfd':fd,'options':flags,
                'actual_kernel_call_returned_without_error':True,
                'event_available':result is not None,'WNOWAIT_child_ownership_witness':True,
                'child_reaped_by_this_witness':False}
        return witness,metadata
    libc=ctypes.CDLL(None,use_errno=True)
    waitid=libc.waitid
    waitid.argtypes=[ctypes.c_int,ctypes.c_uint,ctypes.c_void_p,ctypes.c_int]
    waitid.restype=ctypes.c_int
    # Linux siginfo_t is128 bytes. Do not guess field offsets or interpret event
    # status: a long-aligned opaque buffer is sufficient for the ownership call.
    Buffer=ctypes.c_long*16
    need(ctypes.sizeof(Buffer)==128 and ctypes.alignment(Buffer)>=8,'opaque siginfo buffer ABI')
    metadata['backend']='current_process_libc.waitid_Linux_P_PIDFD3'
    metadata['siginfo_buffer_bytes']=128
    def witness(fd):
        need(type(fd)is int and 0<=fd<2**31,'actual returned pidfd integer')
        info=Buffer();ctypes.set_errno(0)
        rc=waitid(3,fd,ctypes.byref(info),flags);actual_errno=ctypes.get_errno()
        if rc!=0:raise OSError(actual_errno,os.strerror(actual_errno),'libc.waitid(P_PIDFD, actual_owned_pidfd, WNOWAIT)')
        return {'backend':metadata['backend'],'actual_pidfd':fd,'options':flags,
            'actual_libc_return':rc,'actual_errno':actual_errno,
            'actual_kernel_call_returned_without_error':True,
            'WNOWAIT_child_ownership_witness':True,'child_reaped_by_this_witness':False,
            'siginfo_fields_not_interpreted':True,'siginfo_buffer_bytes':128}
    return witness,metadata


def initialize(config, deadline_unix):
    """Source-bound own-stage preimport with original offline dependency lifecycle."""
    need(set(config)=={'owner_request','control_root','expected_font_source','expected_fc_list'},
         'closed actual dependency initialization bindings')
    control=plain(config['control_root'])
    need(control.name.endswith('_DEPENDENCY_BOOTSTRAP') and control.parent.name=='control'
         and control.parents[2]==Path('/root/ours5_b3_full_nuscenes100_20261105_001')
         and control.parent.is_dir() and not os.path.lexists(control),
         'fresh current owned SYS stage dependency namespace')
    need(type(deadline_unix)in(int,float)and math.isfinite(deadline_unix)and time.time()<deadline_unix,
         'actual finite evaluation deadline')
    run_root=control.parents[1]
    qd=config['owner_request']
    need(qd['path']==str(run_root/'inputs/RECEIVED_QUEUE_REQUEST.json'),
         'actual current received queue request path')
    qbody,actual=whole(qd['path'],expected=(qd['bytes'],qd['sha256']))
    need(actual==qd,'actual queue descriptor')
    ownerq=json.loads(qbody)
    need(ownerq['run_root']==str(run_root) and deadline_unix<=ownerq['deadline_unix'],
         'same owned namespace and unextended finite queue deadline')
    pidfd_d=ownerq['pidfd_bridge'];pb,_=whole(pidfd_d['path'],expected=PIDFD_PIN)
    need((pidfd_d['bytes'],pidfd_d['sha256'])==PIDFD_PIN,'unchanged original852f descriptor')
    pidfd=types.ModuleType('_dependency_original852f');pidfd.__file__=pidfd_d['path'];exec(compile(pb,pidfd.__file__,'exec'),pidfd.__dict__)
    waitid_witness,waitid_backend=pidfd_child_waitid_backend()
    font_source,allowed,binary=font_source_commands()
    need(font_source==config['expected_font_source'] and binary==config['expected_fc_list'],
         'current observed installed font source and local binary unchanged')
    control.mkdir(mode=0o700);cache=control/'cache';cache.mkdir(mode=0o700)
    save(control/'INITIALIZATION_INTENT.json',{'schema':'planned-dependency-bootstrap-intent-v1',
        'actual_owner_request':qd,'actual_bootstrap_SOURCE':whole(Path(__file__).resolve())[1],
        'actual_PID':os.getpid(),'actual_OS_argv':Path('/proc/self/cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
         'actual_configured_waitid_backend_not_yet_child_witness':waitid_backend,
        'actual_font_manager_Source':font_source,'source_derived_local_dependency_commands':allowed,
        'actual_local_fontconfig_binary':binary,'own_SYS_cache':str(cache),
        'original_evaluation_deadline_unix':deadline_unix,'no_Torch_model_or_optimizer_or_scene_call':True})
    started=time.monotonic();started_unix=time.time();old_timer=signal.getitimer(signal.ITIMER_REAL)
    old_alarm_deadline=started+old_timer[0]if old_timer[0]>0 else None
    phase_deadline=min(deadline_unix,started_unix+120)
    if old_alarm_deadline is not None:phase_deadline=min(phase_deadline,started_unix+old_timer[0])
    old_env={k:os.environ.get(k)for k in('MPLCONFIGDIR','YOLOV5_CONFIG_DIR')}
    old_socket,old_connect,old_dns=socket.socket,socket.create_connection,socket.getaddrinfo
    extra_dns={k:getattr(socket,k) for k in ('gethostbyname','gethostbyname_ex','gethostbyaddr','getnameinfo') if hasattr(socket,k)}
    original_popen=subprocess.Popen;processes=[];records=[];mode={'defer':False,'pending':[]};error=None
    watched=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM);handlers={s:signal.getsignal(s)for s in watched}
    def blocked(*args,**kwargs):raise RuntimeError('network/DNS forbidden during planned dependency initialization')
    class NoNetworkSocket(old_socket):
        connect=blocked
        connect_ex=blocked
        send=blocked
        sendall=blocked
        sendto=blocked
        sendmsg=blocked
    def stop(sig,_frame):
        mode['pending'].append(sig)
        if not mode['defer']:
            mode['defer']=True
            raise DependencyStop('actual dependency initialization stop signal '+str(sig))
    def deliver():
        if mode['pending']:raise DependencyStop('deferred dependency initialization stop '+str(mode['pending']))
    def observed_popen(*args,**kwargs):
        argv=args[0]if args else kwargs.get('args')
        need(type(argv)in(list,tuple)and list(argv)in[x['argv']for x in allowed]
             and kwargs.get('shell',False)is False and kwargs.get('executable')is None,
             'only actual Source-derived literal local font command; no unknown benign-process exemption')
        env=kwargs.get('env');need(env is None or env.get('PATH')==os.environ.get('PATH'),'dependency PATH unchanged')
        need(Path(shutil.which(argv[0])).resolve(strict=True)==Path(binary['path']),'same actual local executable resolution')
        need(len(processes)<8,'finite Source-derived font dependency process count<=8')
        mode['defer']=True;proc=None
        try:
            proc=original_popen(*args,**kwargs)
            record={'argv':list(argv),'actual_pid':proc.pid,'Popen_returned':True,
                'actual_separate_session_requested':kwargs.get('start_new_session',False),
                'separate_dependency_group_absence_not_claimed':True,'signals':[],'wait_reaped':False}
            item={'process':proc,'pidfd':None,'record':record};processes.append(item);records.append(record)
            fd=pidfd.pidfd_open(proc.pid,0);item['pidfd']=fd
            # WNOWAIT witnesses this pidfd is our actual returned child; no
            # signal is ever based on a guessed/reused numeric PID.
            record['actual_WNOWAIT_kernel_child_witness']=waitid_witness(fd)
            record['pidfd_child_ownership_witness']=True
            record['actual_PROCESS_record']=save(control/('DEPENDENCY_PROCESS_%02d.json'%len(processes)),dict(record))
        finally:mode['defer']=False
        deliver();return proc
    try:
        for sig in watched:signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,max(.001,phase_deadline-time.time()))
        os.environ['MPLCONFIGDIR']=str(cache);os.environ['YOLOV5_CONFIG_DIR']=str(cache)
        socket.socket=NoNetworkSocket;socket.create_connection=blocked;socket.getaddrinfo=blocked
        for k in extra_dns:setattr(socket,k,blocked)
        subprocess.Popen=observed_popen
        # No Torch/model import here. Any package-internal dependency work is
        # recorded and bounded by this actual stage's original owner observer.
        importlib.import_module('matplotlib.pyplot');importlib.import_module('seaborn')
        deliver()
    except BaseException as exc:
        mode['defer']=True;error=exc
    finally:
        mode['defer']=True
        subprocess.Popen=original_popen;socket.socket=old_socket;socket.create_connection=old_connect;socket.getaddrinfo=old_dns
        for k,v in extra_dns.items():setattr(socket,k,v)
        for item in processes:
            proc=item['process'];record=item['record'];fd=item['pidfd']
            try:
                if proc.poll()is None:
                    need(fd is not None,'unknown dependency handle: retain failure, no guessed signal')
                    if error is None:error=RuntimeError('dependency import returned with live child; no benign completion claim')
                    try:
                        pidfd.pidfd_send_signal(fd,signal.SIGTERM,None,0);record['signals'].append('SIGTERM_exact_owned_pidfd')
                    except ProcessLookupError:record['pidfd_target_exited_before_TERM']=True
                    try:code=proc.wait(timeout=min(5,max(.001,deadline_unix-time.time())))
                    except subprocess.TimeoutExpired:
                        if proc.poll()is None:
                            pidfd.pidfd_send_signal(fd,signal.SIGKILL,None,0);record['signals'].append('SIGKILL_exact_owned_pidfd')
                        code=proc.wait(timeout=5)
                else:code=proc.wait(timeout=5)
                record['actual_wait_return']=code;record['wait_reaped']=True
                try:os.kill(proc.pid,0)
                except ProcessLookupError:record['actual_proc_ESRCH_after_wait']=True
                else:record['actual_proc_ESRCH_after_wait']=False
                record['pipe_closed_after_library_read']={k:(getattr(proc,k)is None or getattr(proc,k).closed)for k in('stdin','stdout','stderr')}
                if not all(record['pipe_closed_after_library_read'].values()) and error is None:
                    error=RuntimeError('dependency import returned before its owned read pipes closed')
                if error is not None:
                    for k in('stdin','stdout','stderr'):
                        stream=getattr(proc,k)
                        if stream is not None and not stream.closed:stream.close()
                    record['owned_pipe_close_attempted_during_error_cleanup']=True
                record['actual_EXIT_record']=save(control/('DEPENDENCY_EXIT_%02d.json'%(processes.index(item)+1)),dict(record))
            except BaseException as exc:
                record['closure_error']={'type':type(exc).__name__,'message':str(exc)}
                if error is None:error=exc
            finally:
                if fd is not None:os.close(fd)
        for key,value in old_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        # Keep stop deferral while writing only finite initialization metadata.
        # The original absolute limit is restored with elapsed time deducted.
        signal.setitimer(signal.ITIMER_REAL,0)
    mode['defer']=False
    try:
        modules=[]
        for name in('matplotlib','matplotlib.pyplot','matplotlib.font_manager','seaborn'):
            m=sys.modules.get(name)
            if m is not None and getattr(m,'__file__',None):modules.append({'module':name,'actual_Source':whole(Path(m.__file__).resolve())[1]})
        record={'schema':'planned-dependency-bootstrap-actual-initialization-return-v1',
            'actual_bootstrap_SOURCE':whole(Path(__file__).resolve())[1],'actual_owner_request':qd,
            'actual_PID':os.getpid(),'oneoff_initialization_elapsed_seconds':time.monotonic()-started,
            'actual_dependency_sources':modules,'actual_dependency_children':records,
            'imports_returned':error is None,'network_and_DNS_blocked_during_preimport':True,
            'actual_source_derived_command_registry_not_platform_or_Gate_whitelist':True,
            'all_dependency_waits_zero':all(r.get('actual_wait_return')==0 for r in records),
            'all_returned_dependencies_reaped':all(r.get('wait_reaped')is True for r in records),
            'full_original_error_preserved':None if error is None else {'type':type(error).__name__,'message':str(error)},
            'original_ENV_and_Popen_and_network_functions_restored':True,
            'no_model_optimizer_generator_or_scene_forward_by_this_helper':True,
            'old_target_offline_guard_modified':False,'old_seeded_evaluation_not_started_yet':True}
        rd=save(control/'INITIALIZATION_RETURNED.json',record)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        remaining=(old_alarm_deadline-time.monotonic())if old_alarm_deadline is not None else 0
        signal.setitimer(signal.ITIMER_REAL,max(.001,remaining)if old_alarm_deadline is not None else 0,old_timer[1])
    if error is not None:raise error
    deliver()
    need(all(r['wait_reaped']and r.get('actual_proc_ESRCH_after_wait')is True for r in records),
         'all real dependency children closed, never guessed benign')
    need(time.time()<deadline_unix,'same original deadline after dependency initialization')
    return rd


