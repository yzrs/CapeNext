source script/test_config.sh

for i in $(seq 1 1 5)
do
  work_dir=${work_dir_prefix}/${dir_name}/split${i}
  config_file=${config_prefix}${i}_config.py
  echo "=======================================Split${i}==========================================="
  echo "Split${i} Testing Latest"
  echo "work_dir: ${work_dir} | config_file: ${config_file}"

  if [ ! -f "${work_dir}/latest.pth" ]; then
      echo "模型权重文件 ${work_dir}/latest.pth 不存在"
      continue
  fi
  python test.py ${config_file} "${work_dir}/latest.pth" \
  --cfg-options additional_module_cfg.module_name="${module_name}" | tee "${work_dir}/test_split${i}_random_cls.log"
done