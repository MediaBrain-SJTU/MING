file_path="/mnt/petrelfs/liaoyusheng/oss/test.txt"

# 使用 while 循环来持续检测
while true; do
  if [ -f "$file_path" ]; then
    echo "文件已存在。"
    break  # 如果文件存在，则退出循环
  else
    echo "文件不存在，正在等待..."
    sleep 5  # 每隔5秒检测一次
  fi
done

echo "完成检测。"