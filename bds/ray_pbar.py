import ray
from tqdm import tqdm


class RayProgressBar:
    @staticmethod
    def num_jobs_done_iter(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    @staticmethod
    def show(obj_ids):
        seq = RayProgressBar.num_jobs_done_iter(obj_ids)
        for x in tqdm(seq, total=len(obj_ids)):
            pass

    @staticmethod
    def check():
        assert ray.is_initialized()
