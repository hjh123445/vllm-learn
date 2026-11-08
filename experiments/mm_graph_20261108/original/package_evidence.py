"""Package the final source and completed measurements without dataset images."""
from pathlib import Path
import hashlib
import json
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'


def main():
    acceptance=json.loads((OUT/'completion_suite/acceptance.json').read_text())
    assert acceptance['passed']
    included=[OUT/'load_matrix',OUT/'raw_output_audit.json',OUT/'revision_validation_wire',OUT/'environment.json',
        OUT/'cpu_regression_final.log',OUT/'cpu_regression_transport.log',OUT/'cpu_regression_release.log']
    names=['quality_hf','quality_offline','quality_offline_eager','quality_comparison.json',
        'graph_eager_quality_comparison.json','strict_large','strict_large_release',
        'vision_real_256','acceptance.json','resume_job.json','resume.log',
        'finalize_job.json','finalize.log','graph_eager_quality_and_release','vision_real_job','raw_token_audit']
    included += [OUT/'completion_suite'/name for name in names]
    archive=OUT/'final_benchmark_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in included:
            if p.exists():
                tar.add(p,arcname=p.relative_to(OUT))
        manifest_path=Path('/data/ifly_jyhe23/vllm-learn-graph-audit-20261108/datasets/chartqa/subset/manifest.json')
        tar.add(manifest_path,arcname='dataset/manifest.json')
        model=Path('/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct')
        for name in ('config.json','generation_config.json','preprocessor_config.json',
                'processor_config.json','chat_template.json','model.safetensors.index.json'):
            if (model/name).exists():
                tar.add(model/name,arcname=f'model_metadata/{name}')
        packages=Path('/home/ifly_jyhe23/env/miniconda3/envs/sglang_contrib/lib/python3.12/site-packages')
        references=['transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py',
            'transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py',
            'transformers/vision_utils.py',
            'transformers/image_processing_backends.py',
            'transformers/models/qwen2_vl/image_processing_qwen2_vl.py',
            'vllm/model_executor/models/qwen3_omni_moe_thinker.py',
            'vllm/v1/core/encoder_cache_manager.py']
        for name in references:
            if (packages/name).exists():
                tar.add(packages/name,arcname=f'reference_source/{name}')
    source=OUT/'v9_final_verified_source.zip'
    files=sorted(p for p in (ROOT/'v9').rglob('*.py') if '__pycache__' not in p.parts)
    scripts=['run_completion_suite.py','run_load_matrix.py','run_quality_eager.py',
        'remote_job.py','audit_outputs.py','progress.py','finalize_results.py',
        'validate_real_vision.py','package_evidence.py','build_final_report.py']
    files += [ROOT/name for name in scripts if (ROOT/name).exists()]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,arcname=p.relative_to(ROOT))
        z.writestr('source_sha256.json',json.dumps(manifest,indent=2))
    (OUT/'final_source_manifest.json').write_text(json.dumps(manifest,indent=2))
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (source,archive)}
    (OUT/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2))
    print(json.dumps(artifacts,indent=2))


if __name__=='__main__':
    main()
