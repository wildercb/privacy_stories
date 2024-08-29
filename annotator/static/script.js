document.getElementById('add-question-btn').addEventListener('click', function() {
    const question = prompt("Enter new question");
    if (question) {
        fetch('/add-question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        }).then(response => response.json())
          .then(data => alert(data.message));
    }
});

function toggleCollapse(elementId) {
    const content = document.getElementById(elementId);
    if (content.style.display === "none" || content.style.display === "") {
        document.querySelectorAll('.story-content').forEach(story => {
            story.style.display = 'none';
        });
        content.style.display = "block";
    } else {
        content.style.display = "none";
    }
}
