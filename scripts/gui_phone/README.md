# 创建虚拟环境（默认会在当前目录生成 .venv）
uv venv

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export all_proxy=socks5://127.0.0.1:17891

# uv 默认走 pypi.org，国内很慢，需要指定清华镜像
# export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

uv pip install -r requirements.txt

# 后续启动命令
WEB_PORT=3389 nohup python server.py > server.log 2>&1 &

TODO: 检查为什么用uv启动不行，只能用python启动