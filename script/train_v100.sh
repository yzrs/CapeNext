source script/train_config.sh

for i in "${v100_split_index[@]}"
do
  work_dir=${work_dir_prefix}/${dir_name}/split${i}
  config_file=${config_prefix}${i}_config.py
  echo "=======================================Split${i}==========================================="
  echo "Split${i} Training"
  echo "work_dir: ${work_dir} | config_file: ${config_file}"
  python train.py --config ${config_file} \
    --work-dir ${work_dir} --cfg-options data.samples_per_gpu=${batch_size} data.workers_per_gpu=${workers_per_gpu} \
    additional_module_cfg.module_name="${module_name}"
  if [ ! -f "${work_dir}/latest.pth" ]; then
      echo "模型权重文件 ${work_dir}/latest.pth 不存在"
      continue
  fi
  echo "=======================================Split${i}==========================================="
  echo "Split${i} Testing Latest"
  python test.py ${config_file} "${work_dir}/latest.pth" \
  --cfg-options additional_module_cfg.module_name="${module_name}" | tee "${work_dir}/test_split${i}_latest.log"

#  echo "=======================================Split${i}==========================================="
#  echo "Split${i} Testing Best"
#  # 查找以 best_pck 为前缀的 .pth 文件
#  best_pth_file=$(find "${work_dir}" -type f -name "best_PCK*.pth")
#  if [ $(echo "${best_pth_file}" | wc -l) -ne 1 ]; then
#    echo "错误：未找到唯一匹配的 .pth 文件，请检查工作目录。"
#    exit 1
#  fi
#  echo "best_pth_file: ${best_pth_file}"
#  python test.py ${config_file} "${best_pth_file}" \
#    --cfg-options additional_module_cfg.module_name="${module_name}" | tee "${work_dir}/test_split${i}_best.log"
done