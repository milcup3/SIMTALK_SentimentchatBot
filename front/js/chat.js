const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

let messages = [
  {
    role: "system",
    content: `
당신은 여행지를 추천하기 위해 사용자에게 인터뷰하는 AI입니다.
아래 질문을 순서대로 하나씩 하세요.

❗ 사용자의 응답이 보기와 맞지 않거나 불명확하면, 다음과 같이 다시 질문하세요:
- "죄송해요, 아래 보기 중에서 골라주세요!"
- "조금 더 구체적으로 말씀해 주세요!"

응답이 적절한 경우에만 다음 질문으로 넘어가세요.
6개의 질문이 끝나면 "이제 추천을 시작할게요." 라고 말하고 멈추세요.

질문 목록:
1️⃣ 대륙: 아시아, 유럽, 아메리카, 기타  
2️⃣ 성격: 자연, 도시, 역사, 휴양  
3️⃣ 예산: 저가, 중간, 고급  
4️⃣ 날씨: 더움, 시원함, 눈오는  
5️⃣ 동행자: 혼자, 가족, 연인, 친구  
6️⃣ 교통: 도보, 대중교통, 자가용, 자전거
`
  }
];

let itineraryParts = [];
let placeListText = "";
let currentStep = 0;
let aibMode = false;
let firstPlace = null;
let firstPlaceCoords = null;

function addMessage(text, sender) {
  const msg = document.createElement('div');
  msg.classList.add('message', sender);
  msg.innerHTML = text.replace(/\n/g, "<br>");
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showTypingIndicator() {
  if (document.getElementById('typing-indicator')) return;
  const msg = document.createElement('div');
  msg.classList.add('message', 'ai');
  msg.id = 'typing-indicator';
  msg.innerHTML = `<span class="dot-typing"><span></span><i></i></span> 타이핑 중...`;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

async function sendMessage() {
  const userInput = chatInput.value.trim();
  if (!userInput) return;

  addMessage(userInput, 'user');
  chatInput.value = '';

  if (aibMode) {
    showNextItinerary();
    return;
  }

  showTypingIndicator();
  messages.push({ role: "user", content: userInput });

  const aiReply = await getAIReply(messages);
  removeTypingIndicator();

  if (aiReply) {
    messages.push({ role: "assistant", content: aiReply });
    addMessage(aiReply, 'ai');

    if (aiReply.includes("추천을 시작할게요")) {
      const userAnswers = messages
        .filter(m => m.role === "user")
        .slice(0, 6)
        .map(m => m.content);

      const summary = await summarizeAnswersWithGPT(userAnswers);
      if (summary) {
        addMessage("📦 답변 요약 완료! 감성 여행 추천을 시작할게요.", 'notice');

        showTypingIndicator();
        const recText = await getEmotionalRecommendation(summary);
        removeTypingIndicator();

        if (recText) {
          splitItinerary(recText);
          if (firstPlace) {
            firstPlaceCoords = null;
            getLatLngFromPlace(firstPlace).then(coords => {
              if (coords) firstPlaceCoords = coords;
            });
          }
          aibMode = true;
          currentStep = 0;
          showNextItinerary();
        }
      }
    }
  } else {
    addMessage("❗ AI 응답에 문제가 발생했습니다. 다시 시도해 주세요.", 'ai');
  }
}

async function getAIReply(history) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history }),
  });
  const data = await res.json();
  return data.reply || null;
}

async function summarizeAnswersWithGPT(userAnswers) {
  const sys = {
    role: "system",
    content: "너는 사용자 여행 선호 응답을 보기 항목 중 하나로 요약하는 역할이야. 보기 외에는 절대 사용하지 마."
  };

  const user = {
    role: "user",
    content: `
다음은 사용자의 자유로운 여행 응답입니다. 각 항목을 아래 보기 중 하나로 정리해줘.
보기 외의 표현은 절대 사용하지 마세요. 결과는 JSON 배열로 출력해 주세요.

1️⃣ 대륙: 아시아, 유럽, 아메리카, 기타  
2️⃣ 성격: 자연, 도시, 역사, 휴양  
3️⃣ 예산: 저가, 중간, 고급  
4️⃣ 날씨: 더움, 시원함, 눈오는  
5️⃣ 동행자: 혼자, 가족, 연인, 친구  
6️⃣ 교통: 도보, 대중교통, 자가용, 자전거

사용자 응답:
1. ${userAnswers[0]}
2. ${userAnswers[1]}
3. ${userAnswers[2]}
4. ${userAnswers[3]}
5. ${userAnswers[4]}
6. ${userAnswers[5]}
`
  };

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [sys, user] }),
  });

  const data = await res.json();
  try {
    const jsonText = data.reply.replace(/```(?:json)?/g, "").replace(/```/g, "").trim();
    return JSON.parse(jsonText);
  } catch {
    console.warn("❌ JSON 파싱 실패:", data.reply);
    return null;
  }
}

async function getEmotionalRecommendation(summary) {
  const res = await fetch("/emotional", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ summary })
  });
  return await res.text();
}

function splitItinerary(aibText) {
  const match = aibText.match(/{추천 장소:\s*"([^"]+)"}/);
  if (match) {
    firstPlace = match[1];
  }

  let mainText = aibText.replace(/{추천 장소:[^}]+}/g, "").trim();
  const parts = mainText.split(/(?=\*\*\d+\.\s)/g).map(part => part.trim()).filter(Boolean);
  itineraryParts = parts;
}

function showNextItinerary() {
  if (currentStep < itineraryParts.length) {
    addMessage(itineraryParts[currentStep], 'ai');
    currentStep++;

    if (currentStep === itineraryParts.length) {
      setTimeout(async () => {
        if (placeListText) addMessage(placeListText, 'ai');
        const btn = document.createElement('button');
        btn.textContent = "📍 로드뷰 보기";
        btn.className = 'roadview-btn';
        btn.onclick = async () => {
          let coords = { lat: 37.570841, lng: 126.976891 };
          if (firstPlaceCoords) {
            coords = firstPlaceCoords;
          } else if (firstPlace) {
            const g = await getLatLngFromPlace(firstPlace);
            if (g) coords = g;
          }
          showRoadView(coords.lat, coords.lng);
        };
        chatBox.appendChild(btn);
        chatBox.scrollTop = chatBox.scrollHeight;
      }, 700);
    }
  }
}

async function getLatLngFromPlace(placeName) {
  const url = `/geocode?place=${encodeURIComponent(placeName)}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    if (data.status === "OK" && data.results.length > 0) {
      const loc = data.results[0].geometry.location;
      return { lat: loc.lat, lng: loc.lng };
    }
  } catch (e) {}
  return null;
}

function showRoadView(lat, lng) {
  const container = document.getElementById('roadview-panel');
  container.innerHTML = '';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'roadview-close-btn';
  closeBtn.innerHTML = '&times;';
  closeBtn.onclick = () => {
    container.classList.remove('visible');
    document.getElementById('chat-container').style.marginRight = '0';
  };
  container.appendChild(closeBtn);

  const streetViewDiv = document.createElement('div');
  streetViewDiv.style.width = "100%";
  streetViewDiv.style.height = "100%";
  container.appendChild(streetViewDiv);

  setTimeout(() => {
    container.classList.add('visible');
    document.getElementById('chat-container').style.marginRight = '55%';
  }, 30);

  new google.maps.StreetViewPanorama(streetViewDiv, {
    position: { lat: lat, lng: lng },
    pov: { heading: 165, pitch: 0 },
    zoom: 1
  });
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    if (event.shiftKey) {
      event.preventDefault();
      const start = chatInput.selectionStart;
      const end = chatInput.selectionEnd;
      chatInput.value = chatInput.value.substring(0, start) + "\n" + chatInput.value.substring(end);
      chatInput.selectionStart = chatInput.selectionEnd = start + 1;
    } else {
      event.preventDefault();
      sendMessage();
    }
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  addMessage("🧭 여행지를 추천해드리기 위해 몇 가지 질문을 드릴게요. 하나씩 편하게 답해주세요!", 'notice');

  const aiReply = await getAIReply(messages);
  if (aiReply) {
    messages.push({ role: "assistant", content: aiReply });
    addMessage(aiReply, 'ai');
  } else {
    addMessage("❗ AI 첫 응답에 실패했습니다. 새로고침 해주세요.", 'ai');
  }
});
