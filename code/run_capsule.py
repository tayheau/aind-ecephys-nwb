""" Writes RAW ephys and LFP to an NWB file """

import sys
import argparse
from pathlib import Path
import numpy as np
import os
import json
import time
import logging
import datetime as dt
from datetime import datetime
from uuid import uuid4

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import probeinterface as pi

from neo.rawio import OpenEphysBinaryRawIO

from neuroconv.tools.nwb_helpers import (
    configure_backend,
    get_default_backend_configuration,
)
from neuroconv.tools.spikeinterface.spikeinterface import (
    add_recording_to_nwbfile,
    add_electrodes_info_to_nwbfile,
)

from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

# for NWB Zarr, let's use built-in compressors, so thay can be read without Python
from numcodecs import Blosc

# AIND
try:
    from aind_log_utils import log

    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

from utils import get_devices_from_rig_metadata


# filter and resample LFP
lfp_filter_kwargs = dict(freq_min=0.1, freq_max=500)
lfp_sampling_rate = 2500

# default compressors
default_electrical_series_compressors = dict(hdf5="gzip", zarr=Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE))

# default event line from open ephys
data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

parser = argparse.ArgumentParser(description="Export Ecephys data to NWB")
# positional arguments
backend_group = parser.add_mutually_exclusive_group()
backend_help = "NWB backend. It can be either 'hdf5' or 'zarr'."
backend_group.add_argument(
    "--backend", choices=["hdf5", "zarr"], default="zarr", help=backend_help
)
backend_group.add_argument("static_backend", nargs="?", help=backend_help)


stub_group = parser.add_mutually_exclusive_group()
stub_help = "Write a stub version for testing"
stub_group.add_argument("--stub", action="store_true", help=stub_help)
stub_group.add_argument("static_stub", nargs="?", default="false", help=stub_help)

stub_seconds_group = parser.add_mutually_exclusive_group()
stub_seconds_help = "Duration of stub recording"
stub_seconds_group.add_argument("--stub-seconds", default=10, help=stub_seconds_help)
stub_seconds_group.add_argument("static_stub_seconds", nargs="?", default="10", help=stub_seconds_help)

write_lfp_group = parser.add_mutually_exclusive_group()
write_lfp_help = "Whether to write LFP electrical series"
write_lfp_group.add_argument("--skip-lfp", action="store_true", help=write_lfp_help)
write_lfp_group.add_argument("static_write_lfp", nargs="?", default="true", help=write_lfp_help)

write_raw_group = parser.add_mutually_exclusive_group()
write_raw_help = "Whether to write RAW electrical series"
write_raw_group.add_argument("--write-raw", action="store_true", help=write_raw_help)
write_raw_group.add_argument("static_write_raw", nargs="?", default="false", help=write_raw_help)

lfp_temporal_subsampling_group = parser.add_mutually_exclusive_group()
lfp_temporal_subsampling_help = (
    "Ratio of input samples to output samples in time. Use 0 or 1 to keep all samples. Default is 2."
)
lfp_temporal_subsampling_group.add_argument("--lfp_temporal_factor", default=2, help=lfp_temporal_subsampling_help)
lfp_temporal_subsampling_group.add_argument("static_lfp_temporal_factor", nargs="?", help=lfp_temporal_subsampling_help)

lfp_spatial_subsampling_group = parser.add_mutually_exclusive_group()
lfp_spatial_subsampling_help = (
    "Controls number of channels to skip in spatial subsampling. Use 0 or 1 to keep all channels. Default is 4."
)
lfp_spatial_subsampling_group.add_argument("--lfp_spatial_factor", default=4, help=lfp_spatial_subsampling_help)
lfp_spatial_subsampling_group.add_argument("static_lfp_spatial_factor", nargs="?", help=lfp_spatial_subsampling_help)

lfp_highpass_filter_group = parser.add_mutually_exclusive_group()
lfp_highpass_filter_help = (
    "Cutoff frequency for highpass filter to apply to the LFP recorsings. Default is 0.1 Hz. Use 0 to skip filtering."
)
lfp_highpass_filter_group.add_argument("--lfp_highpass_freq_min", default=0.1, help=lfp_highpass_filter_help)
lfp_highpass_filter_group.add_argument("static_lfp_highpass_freq_min", nargs="?", help=lfp_highpass_filter_help)

# common median referencing for probes in agar
lfp_surface_channel_agar_group = parser.add_mutually_exclusive_group()
lfp_surface_channel_help = "Index of surface channel (e.g. index 0 corresponds to channel 1) of probe for common median referencing for probes in agar. Pass in as JSON string where key is probe and value is surface channel (e.g. \"{'ProbeA': 350, 'ProbeB': 360}\")"
lfp_surface_channel_agar_group.add_argument(
    "--surface_channel_agar_probes_indices", help=lfp_surface_channel_help, default="", type=str
)
lfp_surface_channel_agar_group.add_argument(
    "static_surface_channel_agar_probes_indices", help=lfp_surface_channel_help, nargs="?", type=str
)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")


if __name__ == "__main__":
    t_export_start = time.perf_counter()

    args = parser.parse_args()

    PARAMS = args.params

    if PARAMS is not None:
        try:
            # try to parse the JSON string first to avoid file name too long error
            nwb_ecephys_params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    nwb_ecephys_params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")
        NWB_BACKEND = nwb_ecephys_params.get("backend", "zarr")
        STUB_TEST = nwb_ecephys_params.get("stub", False)
        STUB_SECONDS = float(nwb_ecephys_params.get("stub_seconds", 10))
        WRITE_LFP = nwb_ecephys_params.get("write_lfp", True)
        WRITE_RAW = nwb_ecephys_params.get("write_raw", False)
        TEMPORAL_SUBSAMPLING_FACTOR = int(nwb_ecephys_params.get("lfp_temporal_factor", 2))
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = int(nwb_ecephys_params.get("lfp_spatial_factor", 4))
        HIGHPASS_FILTER_FREQ_MIN = float(nwb_ecephys_params.get("lfp_highpass_freq_min", 0.1))
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = nwb_ecephys_params.get("surface_channel_agar_probes_indices", None)
    else:
        NWB_BACKEND = args.static_backend or args.backend
        stub = args.stub or args.static_stub
        if args.stub:
            STUB_TEST = True
        else:
            STUB_TEST = True if args.static_stub == "true" else False
        STUB_SECONDS = float(args.stub_seconds) or float(args.static_stub_secods)

        if args.skip_lfp:
            WRITE_LFP = False
        else:
            WRITE_LFP = True if args.static_write_lfp == "true" else False

        if args.write_raw:
            WRITE_RAW = True
        else:
            WRITE_RAW = True if args.static_write_raw == "true" else False

        TEMPORAL_SUBSAMPLING_FACTOR = args.static_lfp_temporal_factor or args.lfp_temporal_factor
        TEMPORAL_SUBSAMPLING_FACTOR = int(TEMPORAL_SUBSAMPLING_FACTOR)
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = args.static_lfp_spatial_factor or args.lfp_spatial_factor
        SPATIAL_CHANNEL_SUBSAMPLING_FACTOR = int(SPATIAL_CHANNEL_SUBSAMPLING_FACTOR)
        HIGHPASS_FILTER_FREQ_MIN = args.static_lfp_highpass_freq_min or args.lfp_highpass_freq_min
        HIGHPASS_FILTER_FREQ_MIN = float(HIGHPASS_FILTER_FREQ_MIN)
        SURFACE_CHANNEL_AGAR_PROBES_INDICES = (
            args.static_surface_channel_agar_probes_indices or args.surface_channel_agar_probes_indices
        )
        if SURFACE_CHANNEL_AGAR_PROBES_INDICES != "":
            SURFACE_CHANNEL_AGAR_PROBES_INDICES = json.loads(SURFACE_CHANNEL_AGAR_PROBES_INDICES)
        else:
            SURFACE_CHANNEL_AGAR_PROBES_INDICES = None

    # Use CO_CPUS/SLURM_CPUS_ON_NODE env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    job_kwargs = dict(n_jobs=N_JOBS, progress_bar=False, mp_context="spawn")
    si.set_global_job_kwargs(**job_kwargs)

    if HAVE_AIND_LOG_UTILS:
        # find raw data
        ecephys_folders = [
            p
            for p in data_folder.iterdir()
            if p.is_dir()
            and ("ecephys" in p.name or "behavior" in p.name)
            and ("sorted" not in p.name and "nwb" not in p.name)
            and "ecephys_clipped" not in p.name
        ]
        ecephys_session_folder = ecephys_folders[0]
        session_name = ecephys_session_folder.name
        # look for subject.json and data_description.json files
        subject_json = ecephys_session_folder / "subject.json"
        subject_id = "undefined"
        if subject_json.is_file():
            subject_data = json.load(open(subject_json, "r"))
            subject_id = subject_data["subject_id"]

        data_description_json = ecephys_session_folder / "data_description.json"
        session_name = "undefined"
        if data_description_json.is_file():
            data_description = json.load(open(data_description_json, "r"))
            session_name = data_description["name"]

        log.setup_logging(
            "NWB Packaging Ecephys",
            subject_id=subject_id,
            asset_name=session_name,
        )
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info("\n\nNWB EXPORT ECEPHYS")

    logging.info(f"Running NWB conversion with the following parameters:")
    logging.info(f"Stub test: {STUB_TEST}")
    logging.info(f"Stub seconds: {STUB_SECONDS}")
    logging.info(f"Write LFP: {WRITE_LFP}")
    logging.info(f"Write RAW: {WRITE_RAW}")
    logging.info(f"Temporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
    logging.info(f"Spatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
    logging.info(f"Highpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
    logging.info(f"Surface channel indices for agar probes: {SURFACE_CHANNEL_AGAR_PROBES_INDICES}")

    # find base NWB file
    nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    nwbfile_input_path = None
    if len(nwb_files) == 1:
        nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path is not None:
        logging.info(f"Found NWB file: {nwbfile_input_path}. Setting up NWB backend based on input file type.")
        if nwbfile_input_path.is_dir():
            assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
            NWB_BACKEND = "zarr"
        else:
            NWB_BACKEND = "hdf5"

    logging.info(f"NWB backend: {NWB_BACKEND}")
    if NWB_BACKEND == "zarr":
        io_class = NWBZarrIO
    else:
        io_class = NWBHDF5IO

    job_json_files = [p for p in data_folder.glob('**/*.json') if "job" in p.name]
    job_dicts = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    logging.info(f"Found {len(job_dicts)} JSON job files")

    # check for timestamps to overwrite recording timestamps
    timestamps_folder = data_folder / "timestamps"

    # we create a result NWB file for each experiment/recording
    session_names = np.unique([job_dict["session_name"] for job_dict in job_dicts])

    for session_name in session_names:
        logging.info(f"Session: {session_name}")
        # filter job_dicts for this session
        job_dicts = [jd for jd in job_dicts if jd["session_name"] == session_name]
        input_folder = job_dicts[0].get("input_folder")

        recording_names = [job_dict["recording_name"] for job_dict in job_dicts]        

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        stream_names = []
        for recording_name in recording_names:
            if "group" in recording_name:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-2]
                stream_name = "_".join(recording_name.split("_")[1:-2])
            else:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-1])

            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
        # note: in case of groups, we will need to aggregate the data for each stream into a single recording
        streams_to_process = []
        for stream_name in stream_names:
            # Skip NI-DAQ
            if "NI-DAQ" in stream_name:
                continue
            # LFP are handled later
            if "LFP" in stream_name:
                continue
            streams_to_process.append(stream_name)

        block_ids = sorted(block_ids)
        recording_ids = sorted(recording_ids)
        streams_to_process = sorted(streams_to_process)

        logging.info(f"Number of NWB files to write: {len(block_ids) * len(recording_ids)}")

        logging.info(f"Number of streams to write for each file: {len(streams_to_process)}")

        # Construct 1 nwb file per experiment - streams are concatenated!
        nwb_output_files = []
        electrical_series_to_configure = []
        nwb_output_files = []
        for block_index, block_str in enumerate(block_ids):
            for segment_index, recording_str in enumerate(recording_ids):
                # add recording/experiment id if needed
                nwbfile = None
                read_io = None
                if nwbfile_input_path is not None:
                    nwb_original_file_name = nwbfile_input_path.stem
                    if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                        nwb_file_name = nwb_original_file_name
                    else:
                        nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}"
                    read_io = io_class(str(nwbfile_input_path), "r")
                    logging.info(f"Using existing NWB file: {nwb_file_name}")
                    nwbfile = read_io.read()
                else:
                    # if no input file, create a new one
                    nwb_file_name = f"{session_name}_{block_str}_{recording_str}"

                        
                    if input_folder is not None:
                        try:
                            from aind_nwb_utils.utils import create_base_nwb_file
                            nwbfile = create_base_nwb_file(Path(input_folder))
                        except:
                            logging.info(f"Failed to create base NWB file from metadata.")

                    if nwbfile is None:
                        from pynwb.testing.mock.file import mock_Subject
                        logging.info(f"Creating NWB file with info.")
                        
                        subject = mock_Subject()
                        timezone_info = datetime.now(dt.timezone.utc).astimezone().tzinfo
                        session_start_date_time = datetime.now().replace(
                            tzinfo=timezone_info
                        )
                        institution = None
                        session_id = session_name
                        asset_name = session_id

                        # Store and write NWB file
                        nwbfile = NWBFile(
                            session_description="NWB file generated by AIND pipeline",
                            identifier=str(uuid4()),
                            session_start_time=session_start_date_time,
                            institution=institution,
                            subject=subject,
                            session_id=session_id,
                        )

                # add suffix for stub test
                if STUB_TEST:
                    nwb_file_name = f"{nwb_file_name}_stub"

                nwbfile_output_path = results_folder / f"{nwb_file_name}.nwb"

                # Find probe devices (this will only work for AIND)
                add_probe_device_from_rig = False
                devices_from_rig, target_locations = None, None
                if input_folder is not None:
                    devices_from_rig, target_locations = get_devices_from_rig_metadata(
                        input_folder,
                        segment_index=segment_index
                    )

                probe_device_names = []
                for stream_index, stream_name in enumerate(streams_to_process):
                    recording_name = f"{block_str}_{stream_name}_{recording_str}"
                    logging.info(f"Processing {recording_name}")

                    # load JSON and recordings
                    # we need lists because multiple groups are saved to different JSON files
                    recording_job_dicts = []
                    for job_dict in job_dicts:
                        if recording_name in job_dict["recording_name"]:
                            recording_job_dicts.append(job_dict)

                    recording_lfp = None
                    recordings = []
                    recordings_lfp = []
                    logging.info(f"\tLoading {recording_name} from {len(recording_job_dicts)} JSON files")
                    if len(recording_job_dicts) > 1:
                        # in case of multiple groups, sort by group names
                        sort_idxs = np.argsort([jd["recording_name"] for jd in recording_job_dicts])
                        recording_job_dicts_sorted = np.array(recording_job_dicts)[sort_idxs]
                    else:
                        recording_job_dicts_sorted = recording_job_dicts
                    for recording_job_dict in recording_job_dicts_sorted:
                        recording = si.load(recording_job_dict["recording_dict"], base_folder=data_folder)
                        recording_name = recording_job_dict["recording_name"]
                        skip_times = recording_job_dict.get("skip_times", False)
                        if skip_times:
                            recording.reset_times()
                        timestamps_file = timestamps_folder / f"{recording_name}.npy"
                        if timestamps_file.is_file():
                            logging.info(f"\tSetting synced timestamps from {timestamps_file}")
                            timestamps = np.load(timestamps_file)
                            recording.set_times(timestamps)
                        recordings.append(recording)

                        logging.info(f"\t\t{recording_job_dict['recording_name']}")
                        if "recording_lfp_dict" in recording_job_dict:
                            logging.info(f"\tLoading associated LFP recording")
                            recording_lfp = si.load(recording_job_dict["recording_lfp_dict"], base_folder=data_folder)
                            if skip_times:
                                recording_lfp.reset_times()
                            timestamps_file_lfp = timestamps_folder / f"{recording_name}_lfp.npy"
                            if timestamps_file_lfp.is_file():
                                logging.info(f"\tSetting synced LFP timestamps from {timestamps_file_lfp}")
                                timestamps_lfp = np.load(timestamps_file_lfp)
                                recording_lfp.set_times(timestamps_lfp)
                            recordings_lfp.append(recording_lfp)
                            logging.info(f"\t\t{recording_lfp}")

                    # for multiple groups, aggregate channels
                    if len(recording_job_dicts_sorted) > 1:
                        logging.info(f"\t\tAggregating channels from {len(recordings)} groups")
                        recording = si.aggregate_channels(recordings)
                        # probes_info get lost in aggregation, so we need to manually set them
                        recording.annotate(
                            probes_info=recordings[0].get_annotation("probes_info")
                        )
                        # remove aggregation key property, since it causes typing issue in NWB export
                        if "aggregation_key" in recording.get_property_keys():
                            recording.delete_property("aggregation_key")
                        if len(recordings_lfp) > 0:
                            recording_lfp = si.aggregate_channels(recordings_lfp)
                            recording_lfp.annotate(
                                probes_info=recordings_lfp[0].get_annotation("probes_info")
                            )
                            if "aggregation_key" in recording_lfp.get_property_keys():
                                recording_lfp.delete_property("aggregation_key")

                    if STUB_TEST:
                        end_frame = int(STUB_SECONDS * recording.sampling_frequency)
                        recording = recording.frame_slice(start_frame=0, end_frame=end_frame)
                        if recording_lfp is not None:
                            end_frame = int(STUB_SECONDS * recording_lfp.sampling_frequency)
                            recording_lfp = recording_lfp.frame_slice(start_frame=0, end_frame=end_frame)

                    # Add device and electrode group
                    probe_device_name = None
                    if devices_from_rig:
                        for device_name, device in devices_from_rig.items():
                            # find probe device name associated to stream
                            probe_no_spaces = device_name.replace(" ", "")
                            if probe_no_spaces in stream_name:
                                probe_device_name = device_name
                                electrode_group_location = target_locations.get(device_name, "unknown")
                                logging.info(
                                    f"Found device from rig: {probe_device_name} at location {electrode_group_location}"
                                )
                                add_probe_device_from_rig = True
                                break

                    probe_info = None
                    fixed_probe_device_name = None
                    if probe_device_name is None:
                        electrode_group_location = "unknown"
                        probes_info = recording.get_annotation("probes_info", None)
                        if probes_info is not None and len(probes_info) == 1:
                            probe_info = probes_info[0]
                    else:
                        # deal with Quad Base: the rig.json has the same name for the different shanks
                        # but we have to load the single-shank probe device name
                        probes_info = recording.get_annotation("probes_info", None)
                        if probes_info is not None and len(probes_info) == 1:
                            probe_info = probes_info[0]
                            model_name = probe_info.get("model_name")
                            model_description = probe_info.get("description")
                            is_quad_base = False
                            if model_name is not None and "Quad Base" in model_name:
                                is_quad_base = True
                            elif model_description is not None and "Quad Base" in model_description:
                                is_quad_base = True
                            if is_quad_base:
                                logging.info(f"Detected Quade Base: changing name from {probe_device_name} to {probe_info['name']}")
                                fixed_probe_device_name = {probe_device_name: probe_info["name"]}

                    if probe_info is not None and not add_probe_device_from_rig:
                        probe_device_name = probe_info.get("name", None)
                        probe_device_manufacturer = probe_info.get("manufacturer", None)
                        probe_model_name = probe_info.get("model_name", None)
                        probe_serial_number = probe_info.get("serial_number", None)
                        probe_description = probe_info.get("description", None)
                        probe_device_description = ""
                        probe_device_name = probe_device_name or probe_model_name or "Probe"

                        if probe_model_name is not None:
                            probe_device_description += f"Model: {probe_device_description}"
                        if probe_serial_number is not None:
                            if len(probe_device_description) > 0:
                                probe_device_description += " - "
                            probe_device_description += f"Serial number: {probe_serial_number}"
                        if probe_description is not None:
                            if len(probe_device_description) > 0:
                                probe_device_description += " - "
                            probe_device_description += f"Description: {probe_description}"
                        # this is needed to account for a case where multiple streams have the same device name
                        if len(streams_to_process) > 1 and probe_device_name in probe_device_names:
                            probe_device_name = f"{probe_device_name}-{stream_index}"
                        probe_device = Device(
                            name=probe_device_name,
                            description=probe_device_description,
                            manufacturer=probe_device_manufacturer,
                        )
                        if probe_device_name not in probe_device_names:
                            logging.info(f"\tAdding probe device: {probe_device.name} from recording metadata")

                    if add_probe_device_from_rig:
                        probe_device = devices_from_rig[probe_device_name]
                        if fixed_probe_device_name is not None:
                            probe_device.name = fixed_probe_device_name[probe_device_name]
                        if probe_device.name not in probe_device_names:
                            logging.info(f"\tAdding probe device: {probe_device.name} from asset metadata")

                    # last resort: could not find a device
                    if probe_device_name is None:
                        probe_device_name = "Device"
                        if len(streams_to_process) > 1 and probe_device_name in probe_device_names:
                            probe_device_name = f"{probe_device_name}-{stream_index}"
                        probe_device = Device(name=probe_device_name, description="Default device")
                        if probe_device.name not in probe_device_names:
                            logging.info(f"\tCould not load device information: adding default probe device")

                    # keep track of all added probe device names
                    if probe_device_name not in probe_device_names:
                        nwbfile.add_device(probe_device)
                        probe_device_names.append(probe_device_name)

                    # add other devices (e.g., lasers)
                    if devices_from_rig:
                        for device_name, device in devices_from_rig.items():
                            # skip fixed device names (already added)
                            if fixed_probe_device_name is not None and device_name in fixed_probe_device_name:
                                continue
                            # skip other probe devices
                            if any(device_name.replace(" ", "") in s for s in streams_to_process):
                                continue
                            if device_name not in nwbfile.devices:
                                logging.info(f"\tAdding other device: {device_name} from asset metadata")
                                nwbfile.add_device(device)

                    electrode_metadata = dict(
                        Ecephys=dict(
                            Device=[dict(name=probe_device_name)],
                        )
                    )
                    # Add channel properties (group_name property to associate electrodes with group)
                    channel_groups = recording.get_channel_groups()
                    if len(np.unique(channel_groups)) == 1:
                        recording.set_channel_groups([probe_device_name] * recording.get_num_channels())
                        electrode_groups_metadata = [
                            dict(
                                name=probe_device_name,
                                description=f"Recorded electrodes from probe {probe_device_name}",
                                location=electrode_group_location,
                                device=probe_device_name,
                            )
                        ]
                    else:
                        recording.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])
                        channel_groups_unique = np.unique(recording.get_channel_groups())
                        electrode_groups_metadata = [
                            dict(
                                name=group,
                                description=f"Recorded electrodes from group {group}",
                                location=electrode_group_location,
                                device=probe_device_name,
                            )
                            for group in channel_groups_unique
                        ]
                    electrode_metadata["Ecephys"]["ElectrodeGroup"] = electrode_groups_metadata

                    if WRITE_RAW:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}",
                                description=f"Voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}", write_as="raw"
                        )

                        logging.info(f"\tAdding RAW data for stream {stream_name} - segment {segment_index}")
                        add_recording_to_nwbfile(
                            recording=recording,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            always_write_timestamps=True,
                            **add_electrical_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_series_kwargs["es_key"])
                    else:
                        # always add recording electrodes, as they will be used by Units
                        add_electrodes_info_to_nwbfile(recording=recording, nwbfile=nwbfile, metadata=electrode_metadata)

                    if WRITE_LFP:
                        electrical_series_name = f"ElectricalSeries{probe_device_name}-LFP"
                        electrical_series_metadata = {
                            electrical_series_name: dict(
                                name=f"ElectricalSeries{probe_device_name}-LFP",
                                description=f"LFP voltage traces from {probe_device_name}",
                            )
                        }
                        electrode_metadata["Ecephys"].update(electrical_series_metadata)
                        add_electrical_lfp_series_kwargs = dict(
                            es_key=f"ElectricalSeries{probe_device_name}-LFP",
                            write_as="lfp",
                        )

                        if recording_lfp is None:
                            # Wide-band recording: filter and resample LFP
                            logging.info(
                                f"\tAdding LFP data for stream {stream_name} from wide-band signal - segment {segment_index}"
                            )
                            recording_lfp = spre.bandpass_filter(recording, **lfp_filter_kwargs)
                            recording_lfp = spre.resample(recording_lfp, lfp_sampling_rate)
                            recording_lfp = spre.astype(recording_lfp, dtype="int16")

                            # there is a bug in with sample mismatches for the last chunk if num_samples not divisible by chunk_size
                            # the workaround is to discard the last samples to make it "even"
                            if recording.get_num_segments() == 1:
                                recording_lfp = recording_lfp.frame_slice(
                                    start_frame=0,
                                    end_frame=int(
                                        recording_lfp.get_num_samples() // lfp_sampling_rate * lfp_sampling_rate
                                    ),
                                )
                            # set times
                            lfp_period = 1.0 / lfp_sampling_rate
                            for sg_idx in range(recording.get_num_segments()):
                                ts_lfp = (
                                    np.arange(recording_lfp.get_num_samples(sg_idx))
                                    / recording_lfp.sampling_frequency
                                    - recording.get_times(sg_idx)[0]
                                    + lfp_period / 2
                                )
                                recording_lfp.set_times(ts_lfp, segment_index=sg_idx, with_warning=False)
                            save_to_binary = True
                        else:
                            logging.info(f"\tAdding LFP data for {stream_name} from LFP stream - segment {segment_index}")
                            save_to_binary = False
                            # In this case, since LFPs are in a separate stream, we have to reset channel groups
                            channel_groups = recording_lfp.get_channel_groups()
                            if len(np.unique(channel_groups)) == 1:
                                recording_lfp.set_channel_groups([probe_device_name] * recording_lfp.get_num_channels())
                            else:
                                recording_lfp.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])

                        channel_ids = recording_lfp.get_channel_ids()

                        # re-reference only for agar - subtract median of channels out of brain using surface channel index arg
                        # similar processing to allensdk
                        if SURFACE_CHANNEL_AGAR_PROBES_INDICES is not None:
                            if probe_device_name in SURFACE_CHANNEL_AGAR_PROBES_INDICES:
                                logging.info(f"\t\tCommon median referencing for probe {probe_device_name}")
                                surface_channel_index = SURFACE_CHANNEL_AGAR_PROBES_INDICES[probe_device_name]
                                # get indices of channels out of brain including surface channel
                                reference_channel_indices = np.arange(surface_channel_index, len(channel_ids))
                                reference_channel_ids = channel_ids[reference_channel_indices]
                                # common median reference to channels out of brain
                                recording_lfp = spre.common_reference(
                                    recording_lfp,
                                    reference="global",
                                    ref_channel_ids=reference_channel_ids,
                                )
                            else:
                                logging.info(f"Could not find {probe_device_name} in surface channel dictionary")

                        # spatial subsampling from allensdk - keep every nth channel
                        if SPATIAL_CHANNEL_SUBSAMPLING_FACTOR > 1:
                            logging.info(f"\t\tSpatial subsampling factor: {SPATIAL_CHANNEL_SUBSAMPLING_FACTOR}")
                            channel_ids_to_keep = channel_ids[0 : len(channel_ids) : SPATIAL_CHANNEL_SUBSAMPLING_FACTOR]
                            recording_lfp = recording_lfp.select_channels(channel_ids_to_keep)

                        # time subsampling/decimate
                        if TEMPORAL_SUBSAMPLING_FACTOR > 1:
                            logging.info(f"\t\tTemporal subsampling factor: {TEMPORAL_SUBSAMPLING_FACTOR}")
                            recording_lfp_sub = spre.decimate(recording_lfp, TEMPORAL_SUBSAMPLING_FACTOR)
                            for sg_idx in range(recording.get_num_segments()):
                                lfp_times = recording_lfp.get_times(segment_index=sg_idx)
                                recording_lfp_sub.set_times(lfp_times[::TEMPORAL_SUBSAMPLING_FACTOR], segment_index=sg_idx, with_warning=False)
                            recording_lfp = recording_lfp_sub

                        # high pass filter from allensdk
                        if HIGHPASS_FILTER_FREQ_MIN > 0:
                            logging.info(f"\t\tHighpass filter frequency: {HIGHPASS_FILTER_FREQ_MIN}")
                            recording_lfp = spre.highpass_filter(recording_lfp, freq_min=HIGHPASS_FILTER_FREQ_MIN)

                        # For streams without a separate LFP, save to binary to speed up conversion later
                        if save_to_binary:
                            logging.info(f"\tSaving preprocessed LFP to binary")
                            recording_lfp = recording_lfp.save(
                                folder=scratch_folder / f"{recording_name}-LFP", verbose=False, overwrite=True,
                            )

                        logging.info(f"\tAdding LFP recording {recording_lfp}")
                        add_recording_to_nwbfile(
                            recording=recording_lfp,
                            nwbfile=nwbfile,
                            metadata=electrode_metadata,
                            always_write_timestamps=True,
                            **add_electrical_lfp_series_kwargs,
                        )
                        electrical_series_to_configure.append(add_electrical_lfp_series_kwargs["es_key"])

                logging.info(f"Added {len(streams_to_process)} streams")
                logging.info(f"Configuring {NWB_BACKEND} backend")
                backend_configuration = get_default_backend_configuration(nwbfile=nwbfile, backend=NWB_BACKEND)
                es_compressor = default_electrical_series_compressors[NWB_BACKEND]

                for key in backend_configuration.dataset_configurations.keys():
                    if any([es_name in key for es_name in electrical_series_to_configure]) and "timestamps" not in key:
                        backend_configuration.dataset_configurations[key].compression_method = es_compressor
                configure_backend(nwbfile=nwbfile, backend_configuration=backend_configuration)

                logging.info(f"Writing NWB file to {nwbfile_output_path}")
                if NWB_BACKEND == "zarr":
                    write_args = {"link_data": False}
                else:
                    write_args = {}

                t_write_start = time.perf_counter()
                if nwbfile_input_path is not None:
                    # if we have an input file, we read it and write it to the output file
                    export_io = io_class(str(nwbfile_output_path), "w")
                    logging.info(f"749: scr_io: {read_io}, nwbfile: {nwbfile}, args: {write_args}")
                    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                    read_io.close()
                else:
                    # if no input file, we create a new one
                    nwbfile_output_path = results_folder / f"{nwb_file_name}.nwb"
                    # write the nwb file
                    with io_class(str(nwbfile_output_path), "w") as write_io:
                        logging.info(f"757")
                        write_io.write(nwbfile)
                t_write_end = time.perf_counter()
                elapsed_time_write = np.round(t_write_end - t_write_start, 2)
                logging.info(f"Writing time: {elapsed_time_write}s")
                logging.info(f"Done writing {nwbfile_output_path}")
                nwb_output_files.append(nwbfile_output_path)

    t_export_end = time.perf_counter()
    elapsed_time_export = np.round(t_export_end - t_export_start, 2)
    logging.info(f"NWB EXPORT ECEPHYS time: {elapsed_time_export}s")
