document.addEventListener("DOMContentLoaded", function () {
  // Initialize Particles
  initParticles();

  // Elements
  const heroSection = document.getElementById("hero-section");
  const flowSection = document.getElementById("flow-section");
  const teamSection = document.getElementById("team-section");
  const convertSection = document.getElementById("convert-section");
  const convertBtn = document.getElementById("convert-btn");
  const backBtn = document.getElementById("back-btn");

  const uploadFormSection = document.getElementById("upload-form-section");
  const loadingSection = document.getElementById("loading-section");
  const outputSection = document.getElementById("output-section");

  const uploadForm = document.getElementById("upload-form");
  const audioInput = document.getElementById("audio");
  const videoInput = document.getElementById("video");
  const audioFilename = document.getElementById("audio-filename");
  const videoFilename = document.getElementById("video-filename");
  const generateBtn = document.getElementById("generate-btn");
  const outputVideo = document.getElementById("output-video");
  const downloadBtn = document.getElementById("download-btn");
  const newConversionBtn = document.getElementById("new-conversion-btn");

  // Navigation: Show Convert Section
  convertBtn.addEventListener("click", function () {
    heroSection.classList.add("hidden");
    flowSection.classList.add("hidden");
    teamSection.classList.add("hidden");
    convertSection.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Navigation: Back to Home
  backBtn.addEventListener("click", function () {
    convertSection.classList.add("hidden");
    heroSection.classList.remove("hidden");
    flowSection.classList.remove("hidden");
    teamSection.classList.remove("hidden");
    resetForm();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // File Input Handlers
  audioInput.addEventListener("change", function (e) {
    if (e.target.files.length > 0) {
      audioFilename.textContent = e.target.files[0].name;
      checkFilesSelected();
    }
  });

  videoInput.addEventListener("change", function (e) {
    if (e.target.files.length > 0) {
      videoFilename.textContent = e.target.files[0].name;
      checkFilesSelected();
    }
  });

  function checkFilesSelected() {
    if (audioInput.files.length > 0 && videoInput.files.length > 0) {
      generateBtn.disabled = false;
    } else {
      generateBtn.disabled = true;
    }
  }

  // Form Submission
  generateBtn.addEventListener("click", async function (e) {
    e.preventDefault();

    if (!audioInput.files[0] || !videoInput.files[0]) {
      return;
    }

    // Show loading
    uploadFormSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");
    outputSection.classList.add("hidden");

    const formData = new FormData();
    formData.append("video", videoInput.files[0]);
    formData.append("audio", audioInput.files[0]);

    try {
      const response = await fetch("/generate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Processing failed");

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      outputVideo.src = url;
      downloadBtn.href = url;

      // Show result
      loadingSection.classList.add("hidden");
      outputSection.classList.remove("hidden");
    } catch (err) {
      alert("Error: " + err.message);
      loadingSection.classList.add("hidden");
      uploadFormSection.classList.remove("hidden");
    }
  });

  // New Conversion Button
  newConversionBtn.addEventListener("click", function () {
    resetForm();
    outputSection.classList.add("hidden");
    uploadFormSection.classList.remove("hidden");
  });

  function resetForm() {
    audioInput.value = "";
    videoInput.value = "";
    audioFilename.textContent = "Choose File";
    videoFilename.textContent = "Choose File";
    generateBtn.disabled = true;
    outputVideo.src = "";
  }

  // Particles Animation
  function initParticles() {
    const canvas = document.getElementById("particles-canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 80;

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 1;
        this.speedX = Math.random() * 0.5 - 0.25;
        this.speedY = Math.random() * 0.5 - 0.25;
        this.opacity = Math.random() * 0.5 + 0.2;
      }

      update() {
        this.x += this.speedX;
        this.y += this.speedY;

        if (this.x > canvas.width) this.x = 0;
        if (this.x < 0) this.x = canvas.width;
        if (this.y > canvas.height) this.y = 0;
        if (this.y < 0) this.y = canvas.height;
      }

      draw() {
        ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function init() {
      for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
      }
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();
      }

      requestAnimationFrame(animate);
    }

    init();
    animate();

    window.addEventListener("resize", function () {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });
  }
});
