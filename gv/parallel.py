import sys
import itertools as itr

# Global set of workers - initialized when first called a map function
_g_available_workers = None 

def kill_workers():
    from mpi4py import MPI
    all_workers = range(1, MPI.COMM_WORLD.Get_size())
    for worker in all_workers:
        MPI.COMM_WORLD.send(None, dest=worker, tag=666)

def _init():
    global _g_available_workers
    from mpi4py import MPI
    import atexit
    _g_available_workers = set(range(1, MPI.COMM_WORLD.Get_size()))
    atexit.register(kill_workers)

def imap_unordered(f, workloads):
    from mpi4py import MPI
    N = MPI.COMM_WORLD.Get_size() - 1
    if N == 0:
        for res in itr.imap(f, workloads):
            yield res
        return
     
    global _g_available_workers

    for job_index, workload in enumerate(itr.chain(workloads, itr.repeat(None))):
        if workload is None and len(_g_available_workers) == N:
            break

        while not _g_available_workers or workload is None:
            # Wait to receive results
            status = MPI.Status()
            ret = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == 2:
                yield ret['output_data']
                _g_available_workers.add(status.source)
                if len(_g_available_workers) == N:
                    break

        if _g_available_workers and workload is not None:
            dest_rank = _g_available_workers.pop()

            # Send off job
            task = dict(func=f, input_data=workload, job_index=job_index)
            MPI.COMM_WORLD.send(task, dest=dest_rank, tag=10)

def imap(f, workloads):
    from mpi4py import MPI
    N = MPI.COMM_WORLD.Get_size() - 1
    if N == 0:
        for res in itr.imap(f, workloads):
            yield res
        return
    global _g_available_workers

    results = []
    indices = []

    for job_index, workload in enumerate(itr.chain(workloads, itr.repeat(None))):
        if workload is None and len(_g_available_workers) == N:
            break

        while not _g_available_workers or workload is None:
            # Wait to receive results
            status = MPI.Status()
            ret = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == 2:
                results.append(ret['output_data'])
                indices.append(ret['job_index'])
                _g_available_workers.add(status.source)
                if len(_g_available_workers) == N:
                    break

        if _g_available_workers and workload is not None:
            dest_rank = _g_available_workers.pop()

            # Send off job
            task = dict(func=f, input_data=workload, job_index=job_index)
            MPI.COMM_WORLD.send(task, dest=dest_rank, tag=10)

    for i in indices:
        yield results[i]

def worker():
    from mpi4py import MPI
    while True:
        status = MPI.Status()
        ret = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status) 

        if status.tag == 10:
            #print 'R{0}: RECV:d'.format(rank)
            # Workload received
            func = ret['func']
            res = func(ret['input_data'])

            # Done, let's send it back
            #print 'R{0}: SENDING'.format(rank)
            MPI.COMM_WORLD.send(dict(job_index=ret['job_index'], output_data=res), dest=0, tag=2)

        elif status.tag == 666:
            # Kill code
            sys.exit(0)

def main(name=None):
    """
    Main function.

    Example use:

    if gv.parallel.main():
        gv.parallel.map_unordered 
    """
    if name is not None and name != '__main__':
        return False

    from mpi4py import MPI
    rank =  MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        _init()
        return True
    else:
        worker()
        sys.exit(0)
