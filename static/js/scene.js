import * as THREE from "https://cdn.skypack.dev/three@0.152.2";

let scene, camera, renderer, sphere;

function initScene() {
  const canvas = document.getElementById("scene-canvas");
  if (!canvas) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020617);

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.z = 6;

  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);

  const geometry = new THREE.SphereGeometry(1.5, 64, 64);
  const material = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    metalness: 0.6,
    roughness: 0.3,
  });
  sphere = new THREE.Mesh(geometry, material);
  scene.add(sphere);

  const light1 = new THREE.PointLight(0x38bdf8, 1.2);
  light1.position.set(5, 5, 5);
  scene.add(light1);

  const light2 = new THREE.PointLight(0xf97316, 0.8);
  light2.position.set(-5, -3, -5);
  scene.add(light2);

  const ambient = new THREE.AmbientLight(0xffffff, 0.2);
  scene.add(ambient);

  window.addEventListener("resize", onWindowResize);
  animate();
}

function onWindowResize() {
  if (!renderer || !camera) return;
  const canvas = renderer.domElement;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animate() {
  requestAnimationFrame(animate);
  if (sphere) {
    sphere.rotation.y += 0.004;
    sphere.rotation.x += 0.002;
  }
  renderer.render(scene, camera);
}

export function updateSphereForScore(score) {
  if (!sphere) return;
  const fakeColor = new THREE.Color(0xef4444);
  const realColor = new THREE.Color(0x22c55e);
  const color = realColor.lerp(fakeColor, score);
  sphere.material.color = color;
}

window.addEventListener("DOMContentLoaded", initScene);
