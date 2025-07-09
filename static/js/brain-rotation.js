// Global variables
let scene, camera, renderer, brain, controls;

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait a small amount of time to ensure Three.js is fully loaded
    setTimeout(() => {
        if (typeof THREE === 'undefined') {
            console.error('THREE.js is not loaded');
            displayError('Failed to load 3D library');
            return;
        }
        if (typeof THREE.GLTFLoader === 'undefined') {
            console.error('GLTFLoader is not loaded');
            displayError('Failed to load model loader');
            return;
        }
        initBrain();
    }, 100);
});

function displayError(message) {
    console.error('Brain Visualization Error:', message);
    const container = document.getElementById('brain-container');
    if (container) {
        container.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle" style="font-size: 24px; margin-bottom: 10px;"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

function displayLoading(show = true) {
    const container = document.getElementById('brain-container');
    if (!container) return;

    let loadingDiv = container.querySelector('.loading-indicator');
    
    if (show) {
        if (!loadingDiv) {
            loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading-indicator';
            loadingDiv.innerHTML = `
                <i class="fas fa-spinner fa-spin"></i>
                <p>Loading 3D Brain Model...</p>
            `;
            container.appendChild(loadingDiv);
        }
    } else if (loadingDiv) {
        loadingDiv.remove();
    }
}

function initBrain() {
    try {
        console.log('Initializing brain visualization...');
        const container = document.getElementById('brain-container');
        if (!container) {
            throw new Error('Brain container not found');
        }

        // Scene setup
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);
        
        // Camera setup
        const aspect = container.clientWidth / container.clientHeight;
        camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        camera.position.set(0, 0, 5);

        // Renderer setup
        renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Lighting setup
        setupLighting();

        // Controls setup
        setupControls();

        // Load the model
        loadBrainModel();

        // Handle window resize
        window.addEventListener('resize', onWindowResize, false);
        
    } catch (error) {
        console.error('Error in initBrain:', error);
        displayError(`Failed to initialize 3D visualization: ${error.message}`);
    }
}

function setupLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Directional lights
    const frontLight = new THREE.DirectionalLight(0xffffff, 1);
    frontLight.position.set(0, 0, 10);
    scene.add(frontLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
    backLight.position.set(0, 0, -10);
    scene.add(backLight);

    const topLight = new THREE.DirectionalLight(0xffffff, 0.5);
    topLight.position.set(0, 10, 0);
    scene.add(topLight);
}

function setupControls() {
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 3;
        controls.maxDistance = 10;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1;
    }
}

function loadBrainModel() {
    displayLoading(true);

    const loader = new THREE.GLTFLoader();
    const modelPath = window.MODEL_PATH; // Using the path provided by Flask

    console.log('Loading brain model from:', modelPath);

    loader.load(
        modelPath,
        function (gltf) {
            console.log('Model loaded successfully');
            brain = gltf.scene;

            // Center and scale the model
            const box = new THREE.Box3().setFromObject(brain);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            // Adjust scale
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 3 / maxDim;
            brain.scale.multiplyScalar(scale);

            // Center the model
            brain.position.sub(center.multiplyScalar(scale));

            // Add model to scene
            scene.add(brain);
            
            // Hide loading indicator
            displayLoading(false);
            
            // Start animation
            animate();
        },
        function (xhr) {
            // Loading progress
            const percent = xhr.loaded / xhr.total * 100;
            console.log(percent + '% loaded');
        },
        function (error) {
            console.error('Error loading model:', error);
            displayError('Failed to load brain model. Please try refreshing the page.');
        }
    );
}

function onWindowResize() {
    if (!camera || !renderer || !scene) return;

    const container = document.getElementById('brain-container');
    const aspect = container.clientWidth / container.clientHeight;
    
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);

    if (controls) {
        controls.update();
    }

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}