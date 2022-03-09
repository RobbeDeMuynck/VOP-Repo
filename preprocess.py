import nibabel as nib
import numpy as np
import pathlib

path = pathlib.Path(__file__).parent
for mouse in ["M03", "M04", "M05", "M06", "M07", "M08"]:
    d = {}
    for timestamp in ["-001h", "0.25h", "024h"]:
        d[timestamp] = {}

        path_ct = path / f"original/{mouse}_{timestamp}/CT280.img"
        path_organ = path / f"original/{mouse}_{timestamp}/Organ_280.img"
        path_class = path / f"original/{mouse}_{timestamp}/Organ_280.cls"
        if not path_organ.is_file():
            path_organ = path / f"original/{mouse}_{timestamp}/Organ1_280.img"
        if not path_class.is_file():
            path_class = path / f"original/{mouse}_{timestamp}/Organ1_280.cls"

        ct = nib.load(path_ct).get_fdata()
        organ = nib.load(path_organ).get_fdata()
        with open(path_class) as f:
            lines = f.readlines()
        indices = [int(i) for i in lines[1].split("=")[1].split("\n")[0].split("|")[1:]]
        names = lines[2].split("=")[1].split("\n")[0].split("|")[1:]
        means = [np.mean(ct[organ==i]) for i in indices]

        d[timestamp]["ct"] = ct
        d[timestamp]["organ"] = organ
        d[timestamp]["indices"] = {name: idx for name, idx in zip(names, indices)}
        d[timestamp]["means"] = {name: mean for name, mean in zip(names, means)}

    for timestamp in ["-001h", "0.25h", "024h"]:
        offsets = {idx: d[timestamp]["means"][name] - d["-001h"]["means"][name] for name, idx in d["-001h"]["indices"].items()}
        ct = d["-001h"]["ct"]
        organ = d["-001h"]["organ"]
        for idx, offset in offsets.items():
            ct[organ==idx] += offset
        nib.save(nib.Nifti1Image(ct, np.eye(4)), path / f"processed/{mouse}_{timestamp}_CT280.img")
        nib.save(nib.Nifti1Image(organ, np.eye(4)), path / f"processed/{mouse}_{timestamp}_Organ280.img")