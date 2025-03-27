document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    
    if (hamburger) {
        hamburger.addEventListener('click', function() {
            this.classList.toggle('active');
            navLinks.classList.toggle('mobile-active');
        });
    }
    
    // Demo Button Event Listeners
    const demoBtn = document.getElementById('demoBtn');
    const ctaBtn = document.getElementById('ctaBtn');
    
    if (demoBtn) {
        demoBtn.addEventListener('click', function() {
            window.location.href = 'demo.html';
        });
    }
    
    if (ctaBtn) {
        ctaBtn.addEventListener('click', function() {
            window.location.href = 'demo.html';
        });
    }
    
    // Smooth Scrolling for Anchor Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Scroll Animation
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    
    const animateOnScroll = function() {
        animateElements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('animate');
            }
        });
    };
    
    // Run animation check on load
    animateOnScroll();
    
    // Run animation check on scroll
    window.addEventListener('scroll', animateOnScroll);
    
    // Video Play Button
    const playButton = document.querySelector('.play-button');
    
    if (playButton) {
        playButton.addEventListener('click', function() {
            alert('Video playback would start here in the final implementation.');
            // In a real implementation, replace the placeholder with a video player
            // const video = document.createElement('video');
            // video.src = 'demo-video.mp4';
            // video.controls = true;
            // video.autoplay = true;
            // this.parentElement.replaceChild(video, this.parentElement.querySelector('img'));
            // this.remove();
        });
    }
    
    // Navbar Scroll Effect
    const navbar = document.querySelector('.navbar');
    let scrollPosition = window.scrollY;
    
    const scrollFunction = function() {
        const currentScroll = window.scrollY;
        
        if (currentScroll > scrollPosition) {
            // Scrolling down
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // Scrolling up
            navbar.style.transform = 'translateY(0)';
        }
        
        // Add shadow only when not at the top
        if (currentScroll > 50) {
            navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
        } else {
            navbar.style.boxShadow = 'none';
            navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        }
        
        scrollPosition = currentScroll;
    };
    
    window.addEventListener('scroll', scrollFunction);
});