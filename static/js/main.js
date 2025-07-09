// Global variables
let currentFile = null;

// Show error function
function showError(message) {
    const resultContainer = document.getElementById('resultContainer');
    if (resultContainer) {
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

// Show result function
function showResult(data) {
    const resultContainer = document.getElementById('resultContainer');
    const imagePreview = document.getElementById('imagePreview');
    const heatmapPreview = document.getElementById('heatmapPreview');
    const overlayPreview = document.getElementById('overlayPreview');
    const resultText = document.getElementById('resultText');
    const confidenceText = document.getElementById('confidenceText');

    if (resultContainer && imagePreview && resultText && confidenceText) {
        // Display the original image
        imagePreview.innerHTML = `<h4>Original Image</h4><img src="data:image/jpeg;base64,${data.image}" alt="Uploaded MRI">`;
        
        // Display the results
        resultText.innerHTML = `Result: ${data.prediction}`;
        confidenceText.innerHTML = `Confidence: ${data.confidence}%`;
        
        // Show/hide heatmap and overlay based on prediction
        if (data.prediction === 'Tumor' && data.heatmap && data.overlay) {
            heatmapPreview.innerHTML = `<h4>Heat Map</h4><img src="data:image/jpeg;base64,${data.heatmap}" alt="Tumor Heatmap">`;
            overlayPreview.innerHTML = `<h4>Overlay</h4><img src="data:image/jpeg;base64,${data.overlay}" alt="Tumor Overlay">`;
            heatmapPreview.style.display = 'block';
            overlayPreview.style.display = 'block';
        } else {
            heatmapPreview.style.display = 'none';
            overlayPreview.style.display = 'none';
        }
        
        // Show the container
        resultContainer.style.display = 'flex';

        // Add appropriate styling based on the result
        resultText.className = data.prediction.includes('No Tumor') ? 'result negative' : 'result positive';
    }
}

// Handle file upload
function handleFiles(files) {
    if (files && files.length) {
        const file = files[0];
        
        // Validate file type
        if (!file.type.match('image.*')) {
            showError('Please upload an image file (JPEG, PNG)');
            return;
        }

        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showError('File size must be less than 5MB');
            return;
        }

        processFile(file);
    }
}

// Process the uploaded file
function processFile(file) {
    // Clear previous results
    const resultContainer = document.getElementById('resultContainer');
    if (resultContainer) {
        resultContainer.style.display = 'none';
    }
    
    // Show loading state
    const dropArea = document.getElementById('dropArea');
    if (dropArea) {
        dropArea.innerHTML = `
            <div class="loading-container">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing image...</p>
                <div class="progress-bar">
                    <div class="progress"></div>
                </div>
            </div>
        `;
    }

    // Create FormData and upload
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            showResult(data);
        } else {
            showError(data.error || 'An error occurred while processing the image');
        }
    })
    .catch(error => {
        showError('Error uploading or processing the image. Please try again.');
        console.error('Error:', error);
    })
    .finally(() => {
        // Reset the drop area to its original state
        if (dropArea) {
            dropArea.innerHTML = `
                <i class="fas fa-cloud-upload-alt"></i>
                <p>Drag and drop your MRI scan here, or</p>
                <input type="file" id="fileInput" accept="image/*">
                <label for="fileInput">Choose a file</label>
            `;
            
            // Re-attach event listeners
            setupDropArea();
        }
    });
}

// Setup drop area and file input events
function setupDropArea() {
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');

    if (!dropArea || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropArea.classList.add('highlight');
    }

    function unhighlight(e) {
        dropArea.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleChange, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleChange(e) {
    const files = e.target.files;
    handleFiles(files);
}

// On page load
document.addEventListener('DOMContentLoaded', () => {
    // Setup file upload area
    setupDropArea();

    // Setup "Get Started" button
    const getStartedBtn = document.getElementById('getStartedBtn');
    const dropArea = document.getElementById('dropArea');

    if (getStartedBtn && dropArea) {
        getStartedBtn.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Scroll to the drop area with offset
            const offset = 80; // Adjust if you have a fixed header
            const elementPosition = dropArea.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });

            // Add highlight effect to the drop area
            dropArea.classList.add('highlight');
            setTimeout(() => {
                dropArea.classList.remove('highlight');
            }, 1500);

            // Focus on the file input
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                setTimeout(() => {
                    fileInput.focus();
                }, 1000);
            }
        });
    }
});