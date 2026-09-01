import time
from pycloudflared import try_cloudflare

# 8000번 포트 터널링 실행
tunnel = try_cloudflare(port=8000)
print("\n" + "="*50)
print(f"모바일 / PC 공통 접속 주소: {tunnel.tunnel}")
print("="*50 + "\n")
print("※ 웹사이트에 머무는동안은 터미널을 끄지 마세요! (종료하려면 Ctrl + C)")

# 터널이 끊기지 않도록 대기 유지
try:
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("터널을 종료합니다.")