# app.py  — Flask + (자동) ngrok 퍼블릭 URL + QR 코드 생성

import os
import re
import json
import time
import atexit
import socket
import subprocess
import numpy as np
from urllib.request import urlopen, URLError

from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# ===== 기본 설정 =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24).hex()
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 모델 로드 =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pokemon_mobilenetv2_finetuned.keras")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
model = load_model(MODEL_PATH)

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY가 비어 있습니다. LLM 설명/기술 출력 기능이 동작하지 않을 수 있습니다.")

# ===== 클래스/매핑 =====
name_mapping = {
    "Bulbasaur":"이상해씨","Ivysaur":"이상해풀","Venusaur":"이상해꽃","Charmander":"파이리","Charmeleon":"리자드","Charizard":"리자몽",
    "Squirtle":"꼬부기","Wartortle":"어니부기","Blastoise":"거북왕","Caterpie":"캐터피","Metapod":"단데기","Butterfree":"버터플",
    "Weedle":"뿔충이","Kakuna":"딱충이","Beedrill":"독침붕","Pidgey":"구구","Pidgeotto":"피죤","Pidgeot":"피죤투","Rattata":"꼬렛",
    "Raticate":"레트라","Spearow":"깨비참","Fearow":"깨비드릴조","Ekans":"아보","Arbok":"아보크","Pikachu":"피카츄","Raichu":"라이츄",
    "Sandshrew":"모래두지","Sandslash":"고지","Nidoran-f":"니드런♀","Nidorina":"니드리나","Nidoqueen":"니드퀸","Nidoran-m":"니드런♂",
    "Nidorino":"니드리노","Nidoking":"니드킹","Clefairy":"삐삐","Clefable":"픽시","Vulpix":"식스테일","Ninetales":"나인테일",
    "Jigglypuff":"푸린","Wigglytuff":"푸크린","Zubat":"주뱃","Golbat":"골뱃","Oddish":"뚜벅쵸","Gloom":"냄새꼬","Vileplume":"라플레시아",
    "Paras":"파라스","Parasect":"파라섹트","Venonat":"콘팡","Venomoth":"도나리","Diglett":"디그다","Dugtrio":"닥트리오","Meowth":"나옹",
    "Persian":"페르시온","Psyduck":"고라파덕","Golduck":"골덕","Mankey":"망키","Primeape":"성원숭","Growlithe":"가디","Arcanine":"윈디",
    "Poliwag":"발챙이","Poliwhirl":"슈륙챙이","Poliwrath":"강챙이","Abra":"케이시","Kadabra":"윤겔라","Alakazam":"후딘","Machop":"알통몬",
    "Machoke":"근육몬","Machamp":"괴력몬","Bellsprout":"모다피","Weepinbell":"우츠동","Victreebel":"우츠보트","Tentacool":"왕눈해",
    "Tentacruel":"독파리","Geodude":"꼬마돌","Graveler":"데구리","Golem":"딱구리","Ponyta":"포니타","Rapidash":"날쌩마","Slowpoke":"야돈",
    "Slowbro":"야도란","Magnemite":"코일","Magneton":"레어코일","Farfetchd":"파오리","Doduo":"두두","Dodrio":"두트리오","Seel":"쥬쥬",
    "Dewgong":"쥬레곤","Grimer":"질퍽이","Muk":"질뻐기","Shellder":"셀러","Cloyster":"파르셀","Gastly":"고오스","Haunter":"고우스트",
    "Gengar":"팬텀","Onix":"롱스톤","Drowzee":"슬리프","Hypno":"슬리퍼","Krabby":"크랩","Kingler":"킹크랩","Voltorb":"찌리리공",
    "Electrode":"붐볼","Exeggcute":"아라리","Exeggutor":"나시","Cubone":"탕구리","Marowak":"텅구리","Hitmonlee":"시라소몬",
    "Hitmonchan":"홍수몬","Lickitung":"내루미","Koffing":"또가스","Weezing":"또도가스","Rhyhorn":"뿔카노","Rhydon":"코뿌리","Chansey":"럭키",
    "Tangela":"덩쿠리","Kangaskhan":"캥카","Horsea":"쏘드라","Seadra":"시드라","Goldeen":"콘치","Seaking":"왕콘치","Staryu":"별가사리",
    "Starmie":"아쿠스타","Mr.Mime":"마임맨","Scyther":"스라크","Jynx":"루주라","Electabuzz":"에레브","Magmar":"마그마","Pinsir":"쁘사이저",
    "Tauros":"켄타로스","Magikarp":"잉어킹","Gyarados":"갸라도스","Lapras":"라프라스","Ditto":"메타몽","Eevee":"이브이","Vaporeon":"샤미드",
    "Jolteon":"쥬피썬더","Flareon":"부스터","Porygon":"폴리곤","Omanyte":"암나이트","Omastar":"암스타","Kabuto":"투구","Kabutops":"투구푸스",
    "Aerodactyl":"프테라","Snorlax":"잠만보","Articuno":"프리져","Zapdos":"썬더","Moltres":"파이어","Dratini":"미뇽","Dragonair":"신뇽",
    "Dragonite":"망나뇽","Mewtwo":"뮤츠","Mew":"뮤"
}

class_names = ['Abra','Aerodactyl','Alakazam','Arbok','Arcanine','Articuno','Beedrill','Bellsprout',
'Blastoise','Bulbasaur','Butterfree','Caterpie','Chansey','Charizard','Charmander','Charmeleon',
'Clefable','Clefairy','Cloyster','Cubone','Dewgong','Diglett','Ditto','Dodrio','Doduo','Dragonair',
'Dragonite','Dratini','Drowzee','Dugtrio','Eevee','Ekans','Electabuzz','Electrode','Exeggcute',
'Exeggutor','Farfetchd','Fearow','Flareon','Gastly','Gengar','Geodude','Gloom','Golbat','Goldeen',
'Golduck','Golem','Graveler','Grimer','Growlithe','Gyarados','Haunter','Hitmonchan','Hitmonlee',
'Horsea','Hypno','Ivysaur','Jigglypuff','Jolteon','Jynx','Kabuto','Kabutops','Kadabra','Kakuna',
'Kangaskhan','Kingler','Koffing','Krabby','Lapras','Lickitung','Machamp','Machoke','Machop',
'Magikarp','Magmar','Magnemite','Magneton','Mankey','Marowak','Meowth','Metapod','Mew','Mewtwo',
'Moltres','Mr.Mime','Muk','Nidoking','Nidoqueen','Nidoran-f','Nidoran-m','Nidorina','Nidorino',
'Ninetales','Oddish','Omanyte','Omastar','Onix','Paras','Parasect','Persian','Pidgeot','Pidgeotto',
'Pidgey','Pikachu','Pinsir','Poliwag','Poliwhirl','Poliwrath','Ponyta','Porygon','Primeape',
'Psyduck','Raichu','Rapidash','Raticate','Rattata','Rhydon','Rhyhorn','Sandshrew','Sandslash',
'Scyther','Seadra','Seaking','Seel','Shellder','Slowbro','Slowpoke','Snorlax','Spearow','Squirtle',
'Starmie','Staryu','Tangela','Tauros','Tentacool','Tentacruel','Vaporeon','Venomoth','Venonat',
'Venusaur','Victreebel','Vileplume','Voltorb','Vulpix','Wartortle','Weedle','Weepinbell','Weezing',
'Wigglytuff','Zapdos','Zubat']

# ===== 유틸 =====
def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def predict_pokemon(img_path, img_size=(224, 224)):
    """학습과 동일한 MobileNetV2 전처리로 예측"""
    img = load_img(img_path, target_size=img_size, color_mode='rgb')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    eng_name = class_names[idx]
    kor_name = name_mapping.get(eng_name, eng_name)
    conf = float(np.max(preds[0]))
    return eng_name, kor_name, conf

# ===== LLM: 포켓몬 정보 + MOVES 섹션 =====
def get_pokemon_info(kor_name: str) -> str:
    prompt = f"""
아래 형식을 정확히 지켜서 출력하라. 다른 말, 코드블록, 마크다운 금지.

===OVERVIEW===
- 이름 (한글/영문)
- 타입
- 도감 설명 (간단히 2~3문장)
- 강한 상성 포켓몬 3개
- 약한 상성 포켓몬 3개

===MOVES===
- 대표 기술 4~5개: 각 줄은
  "기술명(한글/영문) - 타입/분류(물리·특수·변화)/위력/명중/PP : 한 줄 설명"
  형식으로 써라. 수치가 불확실하면 '알 수 없음'이라고 표기하라.

대상 포켓몬: {kor_name}
다나까체로 작성하라.
""".strip()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"설명을 불러오지 못했사옵니다. (사유: {e})"

def extract_overview_section(text: str) -> str:
    """===OVERVIEW=== ~ ===MOVES=== 사이만 반환"""
    if not text:
        return ""
    m = re.search(r"===OVERVIEW===\s*(.*?)(?:\n===MOVES===|$)", text, flags=re.S)
    return m.group(1).strip() if m else text

def extract_moves_section(text: str) -> str:
    """===MOVES=== 이후 섹션만 반환"""
    if not text:
        return ""
    m = re.search(r"===MOVES===\s*(.*?)(?:\n===[A-Z]|$)", text, flags=re.S)
    return m.group(1).strip() if m else ""

def parse_moves_list(moves_block: str):
    """MOVES 블록에서 실제 기술 라인만 리스트로 변환(불릿/헤더 제거)"""
    if not moves_block:
        return []
    lines = []
    for raw in moves_block.splitlines():
        s = raw.strip().strip('"“”')   # 양쪽 따옴표 제거
        if not s:
            continue
        if "대표 기술" in s:
            continue
        if s.startswith("•"):
            s = s.lstrip("•").strip()
        if s.startswith("-"):
            s = s.lstrip("-").strip()
        if " - " not in s:
            continue
        lines.append(s)
    return lines

# ===== 라우트 =====
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_name=os.path.basename(MODEL_PATH),
        file_url=None,
        eng_name=None,
        kor_name=None,
        confidence=None,
        info_overview="",
        moves_list=[],
        moves_text=""
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("파일이 첨부되지 않았습니다.")
        return redirect(url_for("index"))

    f = request.files["file"]
    if f.filename == "":
        flash("파일명이 비어 있습니다.")
        return redirect(url_for("index"))

    if not allowed_file(f.filename):
        flash("이미지 파일만 업로드 해주세요. (png, jpg, jpeg, webp, bmp)")
        return redirect(url_for("index"))

    fname = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, fname)
    f.save(save_path)

    try:
        eng, kor, conf = predict_pokemon(save_path)
        conf_text = f"{conf*100:.2f}%"
    except Exception as e:
        flash(f"예측 중 오류: {e}")
        return redirect(url_for("index"))

    info_full = get_pokemon_info(kor)
    overview_text = extract_overview_section(info_full)
    moves_text = extract_moves_section(info_full)
    moves_list = parse_moves_list(moves_text)

    return render_template(
        "index.html",
        model_name=os.path.basename(MODEL_PATH),
        file_url=url_for("uploaded_file", filename=fname),
        eng_name=eng,
        kor_name=kor,
        confidence=conf_text,
        info_overview=overview_text,
        moves_list=moves_list,
        moves_text=moves_text
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

# ===== ngrok + QR 자동화 유틸 =====
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def wait_for_ngrok_url(timeout=30):
    """ngrok 로컬 API(127.0.0.1:4040)에서 https 퍼블릭 URL을 기다렸다가 반환"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                data = json.loads(r.read().decode("utf-8"))
                for t in data.get("tunnels", []):
                    url = t.get("public_url", "")
                    if url.startswith("https://"):
                        return url
        except URLError:
            pass
        time.sleep(1)
    return None

def start_ngrok(port=8000, region="ap"):
    """ngrok http 터널 실행(별도 프로세스) 후 public URL 반환"""
    proc = subprocess.Popen(
        ["ngrok", "http", "--region", region, str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    atexit.register(proc.terminate)
    public_url = wait_for_ngrok_url(timeout=30)
    return proc, public_url

def make_qr_png(url, out="ngrok_qr.png"):
    try:
        import qrcode
    except ImportError:
        print("[WARN] qrcode 미설치: pip install 'qrcode[pil]' 로 설치하세요.")
        return
    img = qrcode.make(url)
    img.save(out)
    try:
        img.show()
    except Exception:
        pass
    print(f"[INFO] QR 코드 생성 완료: {out} → {url}")

# ===== 엔트리 포인트 =====
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))

    # 1) 로컬 접속 주소
    local_ip = get_local_ip()
    print(f"[INFO] Local:  http://{local_ip}:{PORT}")

    # 2) ngrok 자동 실행 → 퍼블릭 URL 확보 → QR 생성
    try:
        ngrok_proc, public_url = start_ngrok(PORT, region="ap")  # 아시아 리전
        if public_url:
            print(f"[INFO] Public: {public_url}")
            make_qr_png(public_url, out="ngrok_qr.png")
        else:
            print("[WARN] ngrok URL을 얻지 못했습니다. ngrok 설치/토큰을 확인하세요.")
            print("       (가이드) brew install ngrok/ngrok/ngrok  →  ngrok config add-authtoken '<토큰>'")
    except FileNotFoundError:
        print("[WARN] ngrok 실행 파일을 찾을 수 없습니다. 먼저 설치하세요: brew install ngrok/ngrok/ngrok")

    # 3) Flask 실행 (외부 접근 허용)
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=PORT, debug=False)