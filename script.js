// Basic interactions and animations

document.addEventListener('DOMContentLoaded', () => {
    // Typing effect for subtitle
    const subtitleElement = document.querySelector('.subtitle');
    const textToType = "MSc Cybersecurity Student @ NCI | Aspiring Security Analyst";
    subtitleElement.textContent = ''; // Clear initial text
    let charIndex = 0;

    function typeText() {
        if (charIndex < textToType.length) {
            subtitleElement.textContent += textToType.charAt(charIndex);
            charIndex++;
            setTimeout(typeText, 50); // Typing speed
        } else {
            // Add blinking cursor after typing finishes
            const cursorSpan = document.createElement('span');
            cursorSpan.className = 'cursor';
            cursorSpan.innerHTML = '&nbsp;';
            subtitleElement.appendChild(cursorSpan);
        }
    }
    
    // Start typing effect after a short delay
    setTimeout(typeText, 500);


    // Smooth scrolling for navigation if added later, and general section fade-ins
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.add('fade-in');
        observer.observe(section);
    });
});
