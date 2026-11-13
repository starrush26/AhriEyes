// 구글 번역 위잿 사용
(function () {
    // 1. 구글 번역 기본 상단 배너 및 툴팁 강제 숨김 스타일 주입
    const style = document.createElement('style');
    // 구버전 + 최신버전 구글 번역 상단 툴바 강제 숨김 
    style.innerHTML = `
        .goog-te-banner-frame.skiptranslate,
        .goog-te-banner-frame,
        iframe.VIpgJd-ZVi9od-ORHb-OEVmcd,
        .VIpgJd-ZVi9od-ORHb-OEVmcd.skiptranslate,
        .goog-te-gadget-icon,
        #goog-gt-tt,
        .goog-te-balloon-frame,
        .VIpgJd-ZVi9od-aZ2wEe-wOHMyf {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        body {
            top: 0px !important;
            position: static !important;
        }
        .ahri-lang-container {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(15, 18, 30, 0.85);
            border: 1px solid rgba(255, 105, 180, 0.35);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 13px;
            color: #e2e8f0;
            position: relative;
            z-index: 9999;
        }
        .ahri-lang-btn {
            background: transparent;
            border: none;
            color: #cbd5e1;
            cursor: pointer;
            padding: 3px 6px;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .ahri-lang-btn:hover, .ahri-lang-btn.active {
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.12);
        }
        .ahri-lang-sep {
            color: #475569;
            user-select: none;
        }
        #google_translate_element {
            display: none;
            position: absolute;
            top: 115%;
            right: 0;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }
        #google_translate_element.open {
            display: block;
        }
    `;
    document.head.appendChild(style);

    // 2. [ 🇰🇷 KO / 🇺🇸 EN / 🇨🇳 CN / 🇯🇵 JP / 🌐 OTHER↓ ] UI 렌더링
    function renderLangBar() {
        const target = document.getElementById('ahrieyes-lang-bar');
        if (!target) return;

        // 오프라인(Air-Gapped Live CD) 환경에서는 번역 바 숨김 처리
        if (!navigator.onLine) {
            target.style.display = 'none';
            return;
        }
        // 4개국 국기 SVG 아이콘 정의
        const flags = {
            // 태극기
            ko: `<svg width="16" height="12" viewBox="0 0 36 24" style="border-radius:2px;vertical-align:middle;margin-right:4px;"><rect width="36" height="24" fill="#fff"/><g transform="translate(18 12) rotate(33.7)"><circle r="6" fill="#0047a0"/><path d="M-6 0A6 6 0 0 1 6 0A3 3 0 0 1 0 0A3 3 0 0 0 -6 0Z" fill="#cd2e3a"/><circle cx="-3" r="3" fill="#cd2e3a"/></g><g stroke="#000" stroke-width="1.2"><path d="M5 6.5l3.5-2.5M6.1 8l3.5-2.5M7.2 9.5l3.5-2.5"/><path d="M27.5 4l3.5 2.5M26.4 5.5l3.5 2.5M25.3 7l3.5 2.5"/><path d="M7.2 14.5l3.5 2.5M6.1 16l3.5 2.5M5 17.5l3.5 2.5"/><path d="M25.3 17l3.5-2.5M26.4 18.5l3.5-2.5M27.5 20l3.5-2.5"/></g></svg>`,
            // 성조기
            en: `<svg width="16" height="12" viewBox="0 0 36 24" style="border-radius:2px;vertical-align:middle;margin-right:4px;"><rect width="36" height="24" fill="#b22234"/><path d="M0 3.7h36M0 7.4h36M0 11.1h36M0 14.8h36M0 18.5h36M0 22.2h36" stroke="#fff" stroke-width="1.8"/><rect width="15" height="13" fill="#3c3b6e"/></svg>`,
            // 오성홍기
            cn: `<svg width="16" height="12" viewBox="0 0 36 24" style="border-radius:2px;vertical-align:middle;margin-right:4px;"><rect width="36" height="24" fill="#de2910"/><polygon points="6,3 7.2,6.8 4,4.5 8,4.5 4.8,6.8" fill="#ffde00"/><circle cx="11" cy="2.5" r="0.8" fill="#ffde00"/><circle cx="13" cy="4.5" r="0.8" fill="#ffde00"/><circle cx="13" cy="7.5" r="0.8" fill="#ffde00"/><circle cx="11" cy="9.5" r="0.8" fill="#ffde00"/></svg>`,
            // 일장기
            ja: `<svg width="16" height="12" viewBox="0 0 36 24" style="border-radius:2px;vertical-align:middle;margin-right:4px;"><rect width="36" height="24" fill="#fff"/><circle cx="18" cy="12" r="6" fill="#bc002d"/></svg>`
        };
        // 번역 바 HTML 구조 생성
        target.classList.add('notranslate');
        target.innerHTML = `
            <div class="ahri-lang-container">
                <button type="button" class="ahri-lang-btn" data-lang="ko">${flags.ko}KO</button>
                <span class="ahri-lang-sep">/</span>
                <button type="button" class="ahri-lang-btn" data-lang="en">${flags.en}EN</button>
                <span class="ahri-lang-sep">/</span>
                <button type="button" class="ahri-lang-btn" data-lang="zh-CN">${flags.cn}CN</button>
                <span class="ahri-lang-sep">/</span>
                <button type="button" class="ahri-lang-btn" data-lang="ja">${flags.ja}JP</button>
                <span class="ahri-lang-sep">/</span>
                <button type="button" class="ahri-lang-btn notranslate" id="ahri-other-lang-btn">🌐 OTHER ▼</button>
                <div id="google_translate_element"></div>
            </div>
        `;

        // 4개국 원클릭 전환 이벤트
        target.querySelectorAll('[data-lang]').forEach(btn => {
            btn.addEventListener('click', () => {
                const langCode = btn.getAttribute('data-lang');
                triggerGoogleTranslate(langCode);
                updateActiveBtn(langCode);
                document.getElementById('google_translate_element').classList.remove('open');
            });
        });

        // OTHER↓ 드롭다운 토글
        const otherBtn = document.getElementById('ahri-other-lang-btn');
        const dropdown = document.getElementById('google_translate_element');

        // 평소에는 숨겼다가 OTHER↓ 버튼 클릭 시 드롭다운 열기
        otherBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdown.classList.toggle('open');
        });

        // 드롭다운 내부(언어 선택 박스) 클릭 시 창이 닫히지 않도록 이벤트 전파 차단
        dropdown.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // 드롭다운 외부 클릭 시 닫기
        document.addEventListener('click', () => {
            if (dropdown) dropdown.classList.remove('open');
        });

        // OTHER↓ 드롭다운에서 언어 선택 완료 시 창 자동 닫힘 + 버튼 하이라이트
        dropdown.addEventListener('change', (e) => {
            if (e.target.classList.contains('goog-te-combo')) {
                dropdown.classList.remove('open');
                updateActiveBtn(e.target.value);
            }
        });
    }

    // 3. 활성화된 언어 버튼 업데이트
    function updateActiveBtn(langCode) {
        // 모든 버튼에서 active 클래스 제거
        document.querySelectorAll('.ahri-lang-btn').forEach(b => b.classList.remove('active'));
        const active = document.querySelector(`.ahri-lang-btn[data-lang="${langCode}"]`);

        if (active) { // 4개국 버튼 중 하나가 활성화된 경우
            active.classList.add('active'); // 해당 버튼 활성화
        } else { // 4개국 버튼 외의 언어 선택 시 OTHER 버튼 활성화
            const otherBtn = document.getElementById('ahri-other-lang-btn'); //
            if (otherBtn) otherBtn.classList.add('active'); // OTHER 버튼 활성화
        }
    }

    // 4. 구글 번역 숨김 select 엘리먼트 강제 트리거
    function triggerGoogleTranslate(langCode) {
        if (langCode === 'ko') {
            // 원본(한국어) 복귀
            document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
            document.cookie = `googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; domain=.${location.hostname};`;
            location.reload();
            return;
        }
        // 구글 번역 select 엘리먼트 강제 트리거
        const select = document.querySelector('.goog-te-combo');
        if (select) {
            select.value = langCode; // 언어 코드 설정
            select.dispatchEvent(new Event('change')); // change 이벤트 강제 발생
        }
    }

    // 5. 구글 번역 스크립트 동적 로드 (100개 이상 언어 리스트를 자동으로 생성)
    window.googleTranslateElementInit = function () {
        new google.translate.TranslateElement({
            pageLanguage: 'ko',
            autoDisplay: false
        }, 'google_translate_element');
    };

    // 6. DOM 로드 완료 여부에 맞춰 즉시 또는 이벤트로 실행
    function initAhriTranslator() {
        renderLangBar();

        // 현재 적용된 언어 쿠키를 확인해 버튼 하이라이트(하늘색) 자동 유지
        const match = document.cookie.match(/googtrans=\/ko\/([^;]+)/);
        const currentLang = match ? match[1] : 'ko';
        updateActiveBtn(currentLang);

        // 구글 번역 스크립트 동적 로드
        if (navigator.onLine) {
            const gtScript = document.createElement('script');
            gtScript.src = '//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
            gtScript.async = true;
            document.body.appendChild(gtScript);
        }
    }

    // DOMContentLoaded 이벤트가 아직 발생하지 않은 경우
    if (document.readyState === 'loading') { 
        // DOMContentLoaded 이벤트 발생 시 initAhriTranslator 실행
        document.addEventListener('DOMContentLoaded', initAhriTranslator);
    } else { // 이미 DOMContentLoaded 이벤트가 발생한 경우 즉시 실행
        initAhriTranslator();
    }

    // 중국어 및 일본어 한자 정정 및 개발자 이름(크레딧 + 푸터) + 각 언어별 동기화
    const fixMyGlobalName = () => {
        // 1. 본문 내 중국어 이름 한자 오표기 정정
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while ((node = walker.nextNode())) {
            if (node.nodeValue.includes('李基贤')) node.nodeValue = node.nodeValue.replace(/李基贤/g, '李基铉');
            if (node.nodeValue.includes('李基賢')) node.nodeValue = node.nodeValue.replace(/李基賢/g, '李基鉉');
        }

        // 2. 현재 활성 언어 3중 감지 (구글 select -> html.lang -> 쿠키 순)
        const combo = document.querySelector('.goog-te-combo');
        const htmlLang = document.documentElement.lang;
        const isTranslated = document.documentElement.classList.contains('translated-ltr') || document.documentElement.classList.contains('translated-rtl');
        const cookieMatch = document.cookie.match(/googtrans=\/[^/]+\/([^;]+)/);

        let lang = 'ko';
        if (combo && combo.value) {
            lang = combo.value;
        } else if (isTranslated && htmlLang && htmlLang !== 'ko') {
            lang = htmlLang;
        } else if (cookieMatch && cookieMatch[1]) {
            lang = cookieMatch[1];
        }

        const nameMap = {
            'ko': '이기현',
            'en': 'Gi Hyeon Lee',
            'ja': 'イ・ギヒョン',
            'zh-CN': '李基铉',
            'zh-TW': '李基鉉'
        };
        const targetName = nameMap[lang] || 'Gi Hyeon Lee';

        // 3. 크레딧과 푸터 이름 동시 업데이트
        document.querySelectorAll('.sync-dev-name').forEach(el => {
            if (el.textContent !== targetName) el.textContent = targetName;
        });

        // 4. 영문 표기일 때 중복 괄호 숨김
        document.querySelectorAll('.dev-en-paren').forEach(el => {
            el.style.display = (targetName === 'Gi Hyeon Lee') ? 'none' : 'inline';
        });

        // 5. 기타(OTHER) 언어 선택 시 버튼 텍스트 직접 동기화
        const otherBtn = document.getElementById('ahri-other-lang-btn');
        if (otherBtn) {
            const mainLangs = ['ko', 'en', 'ja', 'zh-CN', 'zh-TW'];
            let targetBtnText = '🌐 OTHER ▼';

            if (!mainLangs.includes(lang)) {
                try {
                    const displayNames = new Intl.DisplayNames([lang], { type: 'language' });
                    let langName = displayNames.of(lang);
                    langName = langName.charAt(0).toUpperCase() + langName.slice(1);
                    targetBtnText = `🌐 OTHER [${langName}] ▼`;
                } catch (e) {
                    targetBtnText = `🌐 OTHER [${lang.toUpperCase()}] ▼`;
                }
            }

            if (otherBtn.textContent !== targetBtnText) {
                otherBtn.textContent = targetBtnText;
            }
        }
    };

    // 0.3초마다 이름 및 버튼 동기화
    setInterval(fixMyGlobalName, 300);

    // 0.3초마다 이름 및 버튼 동기화
    setInterval(fixMyGlobalName, 300);

    // 새로고침 직후에도 쿠키 기준으로 즉시 반영
    fixMyGlobalName();

    // DOM 변경 감지 시 이름 교체 함수 실행
    const nameObserver = new MutationObserver(() => {
        fixMyGlobalName();
    });

    // DOM 및 html 언어 속성 변경 감지 시작
    nameObserver.observe(document.documentElement, {
        childList: true, // DOM 노드 추가/삭제 감지
        subtree: true, // 하위 노드까지 감지
        characterData: true, // 텍스트 노드 변경 감지
        attributes: true, // 속성 변경 감지
        attributeFilter: ['lang', 'class'] // lang 속성과 class 속성 변경만 감지
    });
})();