from typing import Union, List, Dict, Set

import torch
import numpy as np
from deva.inference.object_info import ObjectInfo


class ObjectManager:
    """
    Object (real) IDs are immutable. The same ID always represent the same object.
    Temporary IDs are the positions of each object in the tensor. It changes as objects get removed.
    """
    def __init__(self):
        self.obj_to_tmp_id: Dict[ObjectInfo, int] = {}
        self.tmp_id_to_obj: Dict[int, ObjectInfo] = {}
        self.obj_id_to_obj: Dict[int, ObjectInfo] = {}

        # We keep track of all historical object IDs to avoid collision
        # even removed object IDs stay in this set
        self.all_historical_object_ids: Set[int] = set()
        self.use_long_id = False

    def _recompute_obj_id_to_obj_mapping(self) -> None:
        self.obj_id_to_obj = {obj.id: obj for obj in self.obj_to_tmp_id}

    def add_new_objects(
            self, objects: Union[List[ObjectInfo], ObjectInfo,
                                 List[int]]) -> (List[int], List[int]):
        if not isinstance(objects, list):
            objects = [objects]

        corresponding_tmp_ids = []
        corresponding_obj_ids = []
        for obj in objects:
            if isinstance(obj, int):
                obj = ObjectInfo(id=obj)

            # create new id if id collides with existing ones
            # simple heuristic to determine RGB/0~255; works most of the time except when it doesn't
            count = 0
            new_obj = ObjectInfo(id=obj.id)
            while (new_obj.id in self.all_historical_object_ids) or (self.use_long_id
                                                                     and new_obj.id < 256):
                if self.use_long_id:
                    new_id = np.random.randint(256, 256**3)
                else:
                    new_id = np.random.randint(1, 256)
                new_obj = ObjectInfo(id=new_id)
                count += 1

                if count > 5000:
                    raise ValueError("We cannot find a new ID for this object." +
                                     "Perhaps you should use long ID?")
            new_obj.copy_meta_info(obj)

            # new object
            new_tmp_id = len(self.obj_to_tmp_id) + 1
            self.obj_to_tmp_id[new_obj] = new_tmp_id
            self.tmp_id_to_obj[new_tmp_id] = new_obj
            self.all_historical_object_ids.add(new_obj.id)
            corresponding_tmp_ids.append(new_tmp_id)
            corresponding_obj_ids.append(new_obj.id)

        self._recompute_obj_id_to_obj_mapping()
        assert corresponding_tmp_ids == sorted(corresponding_tmp_ids), 'tmp id assignment bugged'
        return corresponding_tmp_ids, corresponding_obj_ids

    def delete_object(self, obj_ids_to_remove: Union[int, List[int]]) -> None:
        # delete an object or a list of objects
        # re-sort the tmp ids
        if isinstance(obj_ids_to_remove, int):
            obj_ids_to_remove = [obj_ids_to_remove]

        new_tmp_id = 1
        total_num_id = len(self.obj_to_tmp_id)

        local_obj_to_tmp_id = {}
        local_tmp_to_obj_id = {}

        for tmp_iter in range(1, total_num_id + 1):
            obj = self.tmp_id_to_obj[tmp_iter]
            if obj.id not in obj_ids_to_remove:
                local_obj_to_tmp_id[obj] = new_tmp_id
                local_tmp_to_obj_id[new_tmp_id] = obj
                new_tmp_id += 1

        self.obj_to_tmp_id = local_obj_to_tmp_id
        self.tmp_id_to_obj = local_tmp_to_obj_id
        self._recompute_obj_id_to_obj_mapping()

    def purge_inactive_objects(self,
                               max_missed_detection_count: int) -> (bool, List[int], List[int]):
        # remove tmp ids of objects that are removed
        obj_id_to_be_deleted = []
        tmp_id_to_be_deleted = []
        tmp_id_to_keep = []
        obj_id_to_keep = []

        for obj in self.obj_to_tmp_id:
            if obj.poke_count > max_missed_detection_count:
                obj_id_to_be_deleted.append(obj.id)
                tmp_id_to_be_deleted.append(self.obj_to_tmp_id[obj])
            else:
                tmp_id_to_keep.append(self.obj_to_tmp_id[obj])
                obj_id_to_keep.append(obj.id)

        purge_activated = len(obj_id_to_be_deleted) > 0
        if purge_activated:
            self.delete_object(obj_id_to_be_deleted)
        return purge_activated, tmp_id_to_keep, obj_id_to_keep

    def tmp_to_obj_cls(self, mask) -> torch.Tensor:
        # remap tmp id cls representation to the true object id representation
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in self.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        return new_mask

    def get_tmp_to_obj_mapping(self) -> Dict[int, int]:
        # returns the mapping in a dict format for saving it with pickle
        return {obj.id: tmp_id for obj, tmp_id in self.tmp_id_to_obj.items()}

    def realize_dict(self, obj_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
        # turns a dict indexed by obj id into a tensor, ordered by tmp IDs
        output = []
        for _, obj in self.tmp_id_to_obj.items():
            if obj.id not in obj_dict:
                raise NotImplementedError
            output.append(obj_dict[obj.id])
        output = torch.stack(output, dim=0)
        return output

    def make_one_hot(self, cls_mask: torch.Tensor) -> torch.Tensor:
        output = []
        for _, obj in self.tmp_id_to_obj.items():
            output.append(cls_mask == obj.id)
        if len(output) == 0:
            output = torch.zeros((0, *cls_mask.shape), dtype=torch.bool, device=cls_mask.device)
        else:
            output = torch.stack(output, dim=0)
        return output

    def get_current_segments_info(self) -> List[Dict]:
        segments_info = []
        for obj in self.obj_to_tmp_id:
            segments_info.append({
                'category_id': obj.vote_category_id(),
                'id': int(obj.id),
                'score': obj.vote_score(),
            })
        return segments_info

    @property
    def all_obj_ids(self) -> List[int]:
        return [k.id for k in self.obj_to_tmp_id]

    @property
    def num_obj(self) -> int:
        return len(self.obj_to_tmp_id)

    def has_all(self, objects: List[int]) -> bool:
        for obj in objects:
            if obj not in self.obj_to_tmp_id:
                return False
        return True

    def find_object_by_id(self, obj_id) -> ObjectInfo:
        return self.obj_id_to_obj[obj_id]
