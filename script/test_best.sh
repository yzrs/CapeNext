source script/test_config.sh

for i in $(seq 1 1 5)
do
  work_dir=${work_dir_prefix}/${dir_name}/split${i}
  config_file=${config_prefix}${i}_config.py
  echo "=======================================Split${i}==========================================="
  echo "Split${i} Testing Best"
  echo "work_dir: ${work_dir} | config_file: ${config_file}"
  # 查找以 best_pck 为前缀的 .pth 文件
  best_pth_file=$(find "${work_dir}" -type f -name "best_PCK*.pth")
  if [ $(echo "${best_pth_file}" | wc -l) -ne 1 ]; then
    echo "错误：未找到唯一匹配的 .pth 文件，请检查工作目录。"
    exit 1
  fi
  echo "best_pth_file: ${best_pth_file}"
  python test.py ${config_file} "${best_pth_file}" \
    --cfg-options additional_module_cfg.module_name="${module_name}" | tee "${work_dir}/test_split${i}_best.log"
done