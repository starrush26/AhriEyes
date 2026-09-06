// 전역 사운드 관리 객체
window.SoundManager = {
    sounds: {
        scanning: new Audio('/static/sound/sfx_scanning.mp3'),
        stages: [
        new Audio('/static/sound/sfx_stage1.mp3'),
        new Audio('/static/sound/sfx_stage2.mp3'),
        new Audio('/static/sound/sfx_stage3.mp3'),
        new Audio('/static/sound/sfx_stage4.mp3'),
        new Audio('/static/sound/sfx_stage5.mp3')
        ]
    },

    // 1. 스캐닝 시작
    startScanning() {
        const audio = this.sounds.scanning;
        audio.loop = true;
        audio.currentTime = 0;
        audio.volume = 1.0;
        audio.play().catch(err => {
        console.warn('자동 재생 차단:', err);
        });
    },

    // 2. 스캐닝 페이드아웃 중지
    stopScanning(onComplete) {
        const audio = this.sounds.scanning;
        const fadeDuration = 300;
        const stepTime = 30;
        const stepVolume = audio.volume / (fadeDuration / stepTime);

        const fadeInterval = setInterval(() => {
        if (audio.volume > stepVolume) {
            audio.volume -= stepVolume;
        } else {
            clearInterval(fadeInterval);
            audio.pause();
            audio.currentTime = 0;
            audio.volume = 1.0;
            if (typeof onComplete === 'function') {
            onComplete();
            }
        }
        }, stepTime);
    },

    // 3. 판독 결과 사운드 재생
    playResult(score) {
        let stageIndex = Math.floor(score / 20);
        if (stageIndex >= 5) stageIndex = 4;
        if (stageIndex < 0) stageIndex = 0;

        const targetAudio = this.sounds.stages[stageIndex];
        targetAudio.currentTime = 0;
        targetAudio.volume = 1.0;
        targetAudio.play().catch(err => console.error('결과 사운드 재생 실패:', err));
    }
};