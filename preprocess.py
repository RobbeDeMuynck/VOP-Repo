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
            # EXAMPLE for "M03_024h/Organ_280.cls":
            # ClassColors=0 0 0 255|116 161 166 255|0 85 0 255|201 238 255 255|255 170 255 255|0 0 255 255|176 230 241 255|0 130 182 255|71 205 108 255|0 255 0 255|0 255 255 255|56 65 170 255|175 235 186 255
            # ClassIndices=0|1|2|3|4|5|6|7|8|9|10|11|12
            # ClassNames=unclassified|Trachea|Spleen|Bone|Lung|Heart|Stomach|Bladder|Muscle|Tumor|Kidneys|Liver|Intestine
            
            lines = f.readlines()
        indices = [int(i) for i in lines[1].split("=")[1].split("\n")[0].split("|")[1:]]
        names = lines[2].split("=")[1].split("\n")[0].split("|")[1:]
        # Mean pixel intensity value per organ
        means = [np.mean(ct[organ==i]) for i in indices]

        d[timestamp]["ct"] = ct
        d[timestamp]["organ"] = organ
        d[timestamp]["indices"] = {name: idx for name, idx in zip(names, indices)}
        d[timestamp]["means"] = {name: mean for name, mean in zip(names, means)}

    for timestamp in ["-001h", "0.25h", "024h"]:
        # Per timestep: calculate differences of mean organ intensity in comparison with "-001h"
        offsets = {idx: d[timestamp]["means"][name] - d["-001h"]["means"][name] for name, idx in d["-001h"]["indices"].items()}
        ct = d["-001h"]["ct"]
        organ = d["-001h"]["organ"]
        for idx, offset in offsets.items():
            # Add the mean organ intensity offset to the original "-001h" image
            # to get the processed images  
            ct[organ==idx] += offset
        nib.save(nib.Nifti1Image(ct, np.eye(4)), path / f"processed/{mouse}_{timestamp}_CT280.img")
        nib.save(nib.Nifti1Image(organ, np.eye(4)), path / f"processed/{mouse}_{timestamp}_Organ280.img")