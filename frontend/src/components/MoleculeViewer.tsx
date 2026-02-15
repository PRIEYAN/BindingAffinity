/**
 * 3D Molecule Viewer using Three.js
 */

import { useEffect, useRef } from 'react';

interface MoleculeViewerProps {
  pdbId: string;
  structureUrl?: string;
}

export function MoleculeViewer({ pdbId, structureUrl }: MoleculeViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Dynamic import of Three.js
    import('three').then((THREE) => {
      // Scene setup
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a1a);
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(
        75,
        mountRef.current!.clientWidth / mountRef.current!.clientHeight,
        0.1,
        1000
      );
      camera.position.set(0, 0, 50);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(mountRef.current!.clientWidth, mountRef.current!.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      mountRef.current!.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(10, 10, 5);
      scene.add(directionalLight);

      // Load PDB structure
      if (structureUrl) {
        fetch(structureUrl)
          .then((response) => response.text())
          .then((pdbData) => {
            // Parse PDB and create atoms
            const atoms: any[] = [];
            const lines = pdbData.split('\n');

            lines.forEach((line) => {
              if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
                try {
                  const x = parseFloat(line.substring(30, 38));
                  const y = parseFloat(line.substring(38, 46));
                  const z = parseFloat(line.substring(46, 54));
                  if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
                    atoms.push({ x, y, z });
                  }
                } catch (e) {
                  // Skip invalid lines
                }
              }
            });

            // Create spheres for atoms
            atoms.forEach((pos) => {
              const geometry = new THREE.SphereGeometry(0.5, 16, 16);
              const material = new THREE.MeshPhongMaterial({
                color: 0x00ff00,
                emissive: 0x002200,
              });
              const sphere = new THREE.Mesh(geometry, material);
              sphere.position.set(pos.x, pos.y, pos.z);
              scene.add(sphere);
            });

            // Center and scale
            if (atoms.length > 0) {
              const box = new THREE.Box3();
              scene.children.forEach((child: any) => {
                if (child instanceof THREE.Mesh && child.geometry) {
                  box.expandByObject(child);
                }
              });

              const center = box.getCenter(new THREE.Vector3());
              const size = box.getSize(new THREE.Vector3());
              const maxDim = Math.max(size.x, size.y, size.z);
              const scale = 30 / maxDim;

              scene.children.forEach((child: any) => {
                if (child instanceof THREE.Mesh) {
                  child.position.sub(center);
                  child.position.multiplyScalar(scale);
                }
              });
            }
          })
          .catch((error) => {
            console.error('Error loading PDB:', error);
            // Show placeholder
            const geometry = new THREE.SphereGeometry(5, 32, 32);
            const material = new THREE.MeshPhongMaterial({ color: 0x444444 });
            const sphere = new THREE.Mesh(geometry, material);
            scene.add(sphere);
          });
      }

      // Animation
      const animate = () => {
        requestAnimationFrame(animate);
        if (sceneRef.current) {
          sceneRef.current.rotation.y += 0.005;
          renderer.render(sceneRef.current, camera);
        }
      };
      animate();

      // Handle resize
      const handleResize = () => {
        if (!mountRef.current) return;
        camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        if (mountRef.current && renderer.domElement) {
          try {
            mountRef.current.removeChild(renderer.domElement);
          } catch (e) {
            // Already removed
          }
        }
        renderer.dispose();
      };
    }).catch((error) => {
      console.error('Failed to load Three.js:', error);
    });
  }, [pdbId, structureUrl]);

  return (
    <div
      ref={mountRef}
      className="w-full h-full rounded-xl overflow-hidden bg-gray-900"
      style={{ minHeight: '400px' }}
    />
  );
}
