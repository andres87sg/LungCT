import pydicom
import json
import pydicom_seg
import os
import sys, pydicom
import SimpleITK as sitk
import numpy as np
from typing import List
from os.path import isfile, join
from PIL import Image

class MultiSegmentationWriter:
    
    def __init__(
        self,
        metadata,
        inplane_cropping: bool = False,
        skip_empty_slices: bool = True,
        skip_missing_segment: bool = False,
        series_uid: str = "",
        path_to_save: str = "",
    ):
        self._inplane_cropping = inplane_cropping
        self._skip_empty_slices = skip_empty_slices
        self._skip_missing_segment = skip_missing_segment
        self._path_to_save = path_to_save
        self._series_uid = series_uid
        self._metainfo = {}
        with open(metadata) as ifile:
            self._metainfo = json.load(ifile)
        assert isinstance(self._metainfo, dict)

    def get_series_uid(self):
      return self._series_uid

    def save(
        self, segmentation: sitk.Image, source_images: List[pydicom.Dataset], index: int
    ) -> pydicom.Dataset:

        referenced_instance_sequence = []

        for source_image in source_images:
            reference_instance = pydicom.Dataset()
            reference_instance.ReferencedSOPClassUID = source_image.SOPClassUID
            reference_instance.ReferencedSOPInstanceUID = source_image.SOPInstanceUID
            referenced_instance_sequence.append(reference_instance)

        reference_series_uid = source_images[0].SeriesInstanceUID

        template = self.template_from_dcmqi_metainfo(self._metainfo,index)

        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,
            skip_empty_slices=False,  # encode slices with only zeros
            skip_missing_segment=True
        )

        result = writer.write(segmentation, source_images)

        result.ReferencedSeriesSequence = [pydicom.Dataset()]
        result.ReferencedSeriesSequence[0].SeriesInstanceUID = reference_series_uid
        result.ReferencedSeriesSequence[0].ReferencedInstanceSequence = referenced_instance_sequence

        if self._series_uid == "":
            self._series_uid = result.SeriesInstanceUID

        result.SeriesInstanceUID = self._series_uid
        result.ContentCreatorName = "Imexhs"
        result.Manufacturer = "Imexhs"
        result.ManufacturerModelName = "Segmentation"
        result.SoftwareVersions = "0.1"
        result.InstanceNumber = template.InstanceNumber

        result.is_little_endian = True
        result.is_implicit_VR = False

        result.save_as( os.path.join(self._path_to_save , f'segmentation{index}.dcm') )

    def template_from_dcmqi_metainfo(self,metainfo,index):
        
        # Create dataset from provided JSON
        dataset = pydicom.Dataset()
        tags_with_defaults = [
            ("BodyPartExamined", ""),
            ("ClinicalTrialCoordinatingCenterName", ""),
            ("ClinicalTrialSeriesID", ""),
            ("ClinicalTrialTimePointID", ""),
            ("ContentCreatorName", "AI"),
            ("ContentDescription", "Image segmentation"),
            ("ContentLabel", "SEGMENTATION"),
            ("SeriesDescription", "Segmentation"),
            ("InstanceNumber",1),
            ("SeriesNumber", "400"),
        ]
        metainfo["InstanceNumber"] = index + 1
        for tag_name, default_value in tags_with_defaults:
            dataset.__setattr__(tag_name, metainfo.get(tag_name, default_value))

        if len(metainfo["segmentAttributes"]) > 1:
            raise ValueError(
                "Only metainfo.json files written for single-file input are supported"
            )

        if len(metainfo["segmentAttributes"][0]) <= index :
            raise ValueError(f"Segment {index + 1} was not declared in metadata")

        segment_info = metainfo["segmentAttributes"][0][index]
        segment_info["labelID"] = 1
        dataset.SegmentSequence = pydicom.Sequence(
            [pydicom_seg.template._create_segment_dataset(segment_info)]
        )

        return dataset

def create_segmentations(segmentations_raw: [],metadata,dicom_folder,dest_folder):

  segmentations = []
  for segmentation_raw in segmentations_raw:
    segmentation: sitk.Image = sitk.GetImageFromArray(segmentation_raw)    
    segmentations.append(segmentation)

  dicom_series_paths = [join(dicom_folder, f) for f in os.listdir(dicom_folder) if isfile(join(dicom_folder, f))]
  # Paths to an imaging series related to the segmentation

  source_images = [
    pydicom.dcmread(x, stop_before_pixels=True)
    for x in dicom_series_paths
  ]
  source_images.sort(key = lambda x:x.InstanceNumber)

  writer =  MultiSegmentationWriter(
    metadata=metadata,
    inplane_cropping=False,
    skip_empty_slices=False,  # encode slices with only zeros
    skip_missing_segment=True,
    path_to_save = dest_folder,
  )

  for i in range(len(segmentations)):
    if np.amax(segmentations_raw[i]) != 0:
      writer.save(segmentations[i], source_images, i)
  return writer._series_uid
  
""" Example

metadata = "meta.json"
# dicom_folder: carpeta donde estan los archivos dicom originales
# dest_folder: carpeta donde se guardaran las segmentaciones
# Crea y guarda las segmentaciones
create_segmentations([segmentation],metadata,"dicom_folder","dest_folder")

"""