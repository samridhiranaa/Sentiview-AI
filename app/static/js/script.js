function addMessage(text, sender) {
    const chat = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.classList.add(sender);
    msg.innerText = text;

    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
    const chat = document.getElementById("chat-box");

    const typing = document.createElement("div");
    typing.classList.add("typing");
    typing.id = "typing";
    typing.innerText = "Typing...";

    chat.appendChild(typing);
}

function removeTyping() {
    const typing = document.getElementById("typing");
    if (typing) typing.remove();
}

let waitingForFollowUp = false;

async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value;

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    showTyping();

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            message: message,
            is_follow_up: waitingForFollowUp
        })
    });

    const data = await res.json();
    waitingForFollowUp = !!data.follow_up;

    setTimeout(() => {
        removeTyping();
        addMessage(data.response, "bot");
        
        if (data.follow_up) {
            setTimeout(() => {
                addMessage(data.follow_up, "bot");
            }, 500);
        }
    }, 700);
}

window.onload = function() {
    setTimeout(() => {
        addMessage("Hey! I'm your Sentiment Assistant 😊", "bot");
        addMessage("Tell me how you feel about a product!", "bot");
    }, 500);
};
