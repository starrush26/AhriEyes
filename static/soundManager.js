// 전역 사운드 관리 객체
window.SoundManager = {
sounds: {
    // sound폴더에서 mp3 파일을 불러와 Audio 객체로 생성
    scanning: new Audio('./sound/sfx_scanning.mp3'),
    stages: [
      new Audio('./sound/sfx_stage1.mp3'), // 0~20% (매우 안전)
      new Audio('./sound/sfx_stage2.mp3'), // 20~40% (안전)
      new Audio('./sound/sfx_stage3.mp3'), // 40~60% (주의)
      new Audio('./sound/sfx_stage4.mp3'), // 60~80% (위험)
      new Audio('./sound/sfx_stage5.mp3')  // 80~100% (매우 위험)
    ]
},

  // 1. 판독 중 연속음 재생 (루프 지원)
startScanning() {
    const audio = this.sounds.scanning;
    audio.loop = true;
    audio.currentTime = 0;
    audio.volume = 1.0;
    audio.play().catch(err => {
    console.warn('오디오 자동재생 제한 (사용자 인터랙션 필요):', err);
    });
},

  // 2. 판독 완료 시 0.3초 부드러운 페이드아웃 후 정지
stopScanning(onComplete) {
    const audio = this.sounds.scanning;
    const fadeDuration = 300; // 0.3초
    const stepTime = 30;
    const stepVolume = audio.volume / (fadeDuration / stepTime);

    const fadeInterval = setInterval(() => { // 0.3초 동안 볼륨을 점진적으로 줄임
    if (audio.volume > stepVolume) {
        audio.volume -= stepVolume;
    } else { // 볼륨이 0 이하가 되면 정지
        clearInterval(fadeInterval);
        audio.pause();
        audio.currentTime = 0;
        audio.volume = 1.0;

        if (typeof onComplete === 'function') { // onComplete 콜백이 제공되면 호출
            onComplete();
            }
        }
    }, stepTime);
},

  // 3. 점수(0~100)에 맞는 단계별 결과 단발음 출력
playResult(score) {
    let stageIndex = Math.floor(score / 20);
    if (stageIndex >= 5) stageIndex = 4;
    if (stageIndex < 0) stageIndex = 0;

    const targetAudio = this.sounds.stages[stageIndex]; // 해당 점수 단계의 오디오 선택
    targetAudio.currentTime = 0; // 재생 전부 정지
    targetAudio.volume = 1.0; // 볼륨 초기화
    targetAudio.play().catch(err => console.error('결과 사운드 재생 실패:', err)); // 오디오 재생 실패 시 예외처리
    }
};