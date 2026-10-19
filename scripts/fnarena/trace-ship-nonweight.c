#define _GNU_SOURCE
#include <stdint.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
static __thread int busy;
static const char *prefix="/mnt/models/fnship/output/llama-moe-output-";
static int own(const char*p,int deleted){size_t n=strlen(prefix);if(strncmp(p,prefix,n)||strlen(p)<n+6)return 0;for(int i=0;i<6;i++)if(!isalnum((unsigned char)p[n+i]))return 0;return !strcmp(p+n+6,deleted?" (deleted)":"");}
static uint64_t reads(void){char name[96],buf[4096];snprintf(name,sizeof(name),"/proc/self/task/%ld/io",syscall(SYS_gettid));int fd=syscall(SYS_openat,AT_FDCWD,name,O_RDONLY,0);if(fd<0)return UINT64_MAX;ssize_t n=syscall(SYS_read,fd,buf,sizeof(buf)-1);syscall(SYS_close,fd);if(n<=0)return UINT64_MAX;buf[n]=0;char*p=strstr(buf,"\nread_bytes:");return p?strtoull(p+12,0,10):UINT64_MAX;}
static int mapped(void*addr,size_t len,char*path){uintptr_t a=(uintptr_t)addr;if(!len||a>UINTPTR_MAX-len)return 0;FILE*f=fopen("/proc/self/maps","r");if(!f)return 0;char line[8192];int found=0;while(fgets(line,sizeof(line),f)){unsigned long lo,hi,off,inode;char perms[8],dev[32];int used=0;if(sscanf(line,"%lx-%lx %7s %lx %31s %lu %n",&lo,&hi,perms,&off,dev,&inode,&used)!=6)continue;if(a<lo||a+len>hi)continue;char*p=line+used;size_t n=strcspn(p,"\n");if(n>=4096)break;memcpy(path,p,n);path[n]=0;found=own(path,1)&&perms[0]=='r'&&perms[1]=='w'&&perms[3]=='s';break;}fclose(f);return found;}
static void logio(const char*op,const char*path,uint64_t before,struct timespec a,long result){int e=errno;struct timespec b;clock_gettime(CLOCK_REALTIME,&b);uint64_t after=reads();if(before==UINT64_MAX||after==UINT64_MAX||after<before)fprintf(stderr,"FNSHIP_NONWEIGHT_PROBE_ERROR op=%s\n",op);else if(after>before)fprintf(stderr,"FNSHIP_NONWEIGHT_PROBE op=%s start=%lld.%09ld end=%lld.%09ld tid=%ld result=%ld read_bytes=%"PRIu64" path=%s\n",op,(long long)a.tv_sec,a.tv_nsec,(long long)b.tv_sec,b.tv_nsec,syscall(SYS_gettid),result,after-before,path);errno=e;}
int unlink(const char*path){int saved=errno;if(busy)return syscall(SYS_unlink,path);busy=1;struct stat st;int ok=own(path,0)&&syscall(SYS_newfstatat,AT_FDCWD,path,&st,AT_SYMLINK_NOFOLLOW)==0&&S_ISREG(st.st_mode)&&st.st_size==0;struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;errno=saved;int r=syscall(SYS_unlink,path);int e=errno;if(ok)logio("unlink_empty",path,before,a,r);busy=0;errno=e;return r;}
int msync(void*addr,size_t len,int flags){int saved=errno;if(busy)return syscall(SYS_msync,addr,len,flags);busy=1;char path[4096];int ok=mapped(addr,len,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;errno=saved;int r=syscall(SYS_msync,addr,len,flags);int e=errno;if(ok)logio("msync_owned",path,before,a,r);busy=0;errno=e;return r;}
int munmap(void*addr,size_t len){int saved=errno;if(busy)return syscall(SYS_munmap,addr,len);busy=1;char path[4096];int ok=mapped(addr,len,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;errno=saved;int r=syscall(SYS_munmap,addr,len);int e=errno;if(ok)logio("munmap_owned",path,before,a,r);busy=0;errno=e;return r;}

int close(int fd){int saved=errno;if(busy)return syscall(SYS_close,fd);busy=1;char name[64],path[4096];snprintf(name,sizeof(name),"/proc/self/fd/%d",fd);ssize_t n=syscall(SYS_readlink,name,path,sizeof(path)-1);int ok=0;struct stat st;if(n>=0){path[n]=0;ok=own(path,1)&&syscall(SYS_fstat,fd,&st)==0&&S_ISREG(st.st_mode);}struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;errno=saved;int r=syscall(SYS_close,fd);int e=errno;if(ok)logio("close_owned",path,before,a,r);busy=0;errno=e;return r;}

/* Diagnostic only: exact GPU device source -> owned nonweight output mapping.
 * Attribute queries finish before the measured interval. No graph synchronization.
 * No read exemption is granted by this tracer. */
static int d2h_owned(void *dst,const void *src,size_t len,enum cudaMemcpyKind kind,char *path) {
    if(kind!=cudaMemcpyDeviceToHost || !mapped(dst,len,path))return 0;
    typedef cudaError_t (*attr_fn)(struct cudaPointerAttributes*,const void*);
    attr_fn attr=(attr_fn)dlsym(RTLD_NEXT,"cudaPointerGetAttributes");
    struct cudaPointerAttributes a;
    return attr && attr(&a,src)==cudaSuccess && a.type==cudaMemoryTypeDevice;
}
cudaError_t cudaMemcpy(void*dst,const void*src,size_t len,enum cudaMemcpyKind kind){
    typedef cudaError_t (*fn_t)(void*,const void*,size_t,enum cudaMemcpyKind);
    int saved=errno;fn_t fn=(fn_t)dlsym(RTLD_NEXT,"cudaMemcpy");if(!fn){errno=saved;return cudaErrorUnknown;}
    if(busy){errno=saved;return fn(dst,src,len,kind);}busy=1;char path[4096];
    int ok=d2h_owned(dst,src,len,kind,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;
    errno=saved;cudaError_t r=fn(dst,src,len,kind);int e=errno;if(ok)logio("cuda_d2h_owned",path,before,a,r);busy=0;errno=e;return r;
}
cudaError_t cudaMemcpyAsync(void*dst,const void*src,size_t len,enum cudaMemcpyKind kind,cudaStream_t stream){
    typedef cudaError_t (*fn_t)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t);
    int saved=errno;fn_t fn=(fn_t)dlsym(RTLD_NEXT,"cudaMemcpyAsync");if(!fn){errno=saved;return cudaErrorUnknown;}
    if(busy){errno=saved;return fn(dst,src,len,kind,stream);}busy=1;char path[4096];
    int ok=d2h_owned(dst,src,len,kind,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;
    errno=saved;cudaError_t r=fn(dst,src,len,kind,stream);int e=errno;if(ok)logio("cuda_d2h_async_owned",path,before,a,r);busy=0;errno=e;return r;
}
cudaError_t cudaMemcpy2D(void*dst,size_t dpitch,const void*src,size_t spitch,size_t width,size_t height,enum cudaMemcpyKind kind){
    typedef cudaError_t (*fn_t)(void*,size_t,const void*,size_t,size_t,size_t,enum cudaMemcpyKind);
    int saved=errno;fn_t fn=(fn_t)dlsym(RTLD_NEXT,"cudaMemcpy2D");if(!fn){errno=saved;return cudaErrorUnknown;}
    if(busy){errno=saved;return fn(dst,dpitch,src,spitch,width,height,kind);}busy=1;char path[4096];
    size_t len=(height && width && width<=dpitch && (height-1)<=(SIZE_MAX-width)/dpitch)?(height-1)*dpitch+width:0;
    int ok=d2h_owned(dst,src,len,kind,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;
    errno=saved;cudaError_t r=fn(dst,dpitch,src,spitch,width,height,kind);int e=errno;if(ok)logio("cuda_d2h_2d_owned",path,before,a,r);busy=0;errno=e;return r;
}
cudaError_t cudaMemcpy2DAsync(void*dst,size_t dpitch,const void*src,size_t spitch,size_t width,size_t height,enum cudaMemcpyKind kind,cudaStream_t stream){
    typedef cudaError_t (*fn_t)(void*,size_t,const void*,size_t,size_t,size_t,enum cudaMemcpyKind,cudaStream_t);
    int saved=errno;fn_t fn=(fn_t)dlsym(RTLD_NEXT,"cudaMemcpy2DAsync");if(!fn){errno=saved;return cudaErrorUnknown;}
    if(busy){errno=saved;return fn(dst,dpitch,src,spitch,width,height,kind,stream);}busy=1;char path[4096];
    size_t len=(height && width && width<=dpitch && (height-1)<=(SIZE_MAX-width)/dpitch)?(height-1)*dpitch+width:0;
    int ok=d2h_owned(dst,src,len,kind,path);struct timespec a;clock_gettime(CLOCK_REALTIME,&a);uint64_t before=ok?reads():0;
    errno=saved;cudaError_t r=fn(dst,dpitch,src,spitch,width,height,kind,stream);int e=errno;if(ok)logio("cuda_d2h_2d_async_owned",path,before,a,r);busy=0;errno=e;return r;
}
