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

    // Email copy logic
    const emailLinks = document.querySelectorAll('a[href^="mailto:"]');
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);

    emailLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent opening default mail client
            const email = link.getAttribute('href').replace('mailto:', '');
            
            navigator.clipboard.writeText(email).then(() => {
                toast.innerHTML = `<i class="fas fa-check-circle" style="color: var(--accent-color); margin-right: 8px;"></i> Copied <b>${email}</b> to clipboard!`;
                toast.classList.add('show');
                
                setTimeout(() => {
                    toast.classList.remove('show');
                }, 3000);
            }).catch(err => {
                console.error('Failed to copy email: ', err);
                toast.innerHTML = `Email: <b>${email}</b>`;
                toast.classList.add('show');
                setTimeout(() => toast.classList.remove('show'), 4000);
            });
        });
    });
});

