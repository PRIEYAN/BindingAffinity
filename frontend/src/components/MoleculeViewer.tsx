/**
 * Enhanced 3D Molecule Viewer using Three.js with realistic colors and ball-and-stick model
 */

import { useEffect, useRef, useState } from 'react';

interface MoleculeViewerProps {
  pdbId: string;
  structureUrl?: string;
}

// CPK Color scheme for atoms (realistic molecular visualization colors)
const ATOM_COLORS: { [key: string]: number } = {
  'C': 0x909090,  // Carbon - Dark gray
  'H': 0xffffff,  // Hydrogen - White
  'N': 0x3050f8,  // Nitrogen - Blue
  'O': 0xff0d0d,  // Oxygen - Red
  'P': 0xff8000,  // Phosphorus - Orange
  'S': 0xffff30,  // Sulfur - Yellow
  'F': 0x90e050,  // Fluorine - Green
  'CL': 0x1ff01f, // Chlorine - Green
  'BR': 0xa62929, // Bromine - Dark red
  'I': 0x940094,  // Iodine - Purple
  'FE': 0xe06633, // Iron - Orange-brown
  'ZN': 0x7d80b0, // Zinc - Blue-gray
  'CA': 0x3dff00, // Calcium - Green
  'MG': 0x8aff00, // Magnesium - Green
  'NA': 0xab5cf2, // Sodium - Purple
  'K': 0x8f40d4,  // Potassium - Purple
  'CU': 0xc88033, // Copper - Orange
  'MN': 0x9c7ac7, // Manganese - Purple
};

// Van der Waals radii (in Angstroms) for atom sizing
const ATOM_RADII: { [key: string]: number } = {
  'C': 1.7,
  'H': 1.2,
  'N': 1.55,
  'O': 1.52,
  'P': 1.8,
  'S': 1.8,
  'F': 1.47,
  'CL': 1.75,
  'BR': 1.85,
  'I': 1.98,
  'FE': 2.0,
  'ZN': 1.39,
  'CA': 2.0,
  'MG': 1.73,
  'NA': 2.27,
  'K': 2.75,
  'CU': 1.4,
  'MN': 1.79,
};

interface Atom {
  x: number;
  y: number;
  z: number;
  element: string;
  atomName: string;
  residue: string;
}

function getAtomColor(element: string): number {
  const elem = element.trim().toUpperCase();
  return ATOM_COLORS[elem] || 0xff69b4; // Default: Hot pink for unknown
}

function getAtomRadius(element: string): number {
  const elem = element.trim().toUpperCase();
  return ATOM_RADII[elem] || 1.5; // Default radius
}

function parsePDB(pdbData: string): { atoms: Atom[]; bonds: Array<[number, number]> } {
  const atoms: Atom[] = [];
  const lines = pdbData.split('\n');

  lines.forEach((line) => {
    if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
      try {
        const x = parseFloat(line.substring(30, 38));
        const y = parseFloat(line.substring(38, 46));
        const z = parseFloat(line.substring(46, 54));
        
        if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
          // Extract element (column 76-77, or infer from atom name)
          let element = line.substring(76, 78).trim();
          if (!element) {
            // Infer from atom name (column 12-16)
            const atomName = line.substring(12, 16).trim();
            element = atomName.replace(/[0-9]/g, '').substring(0, 2);
          }
          
          const residue = line.substring(17, 20).trim();
          const atomName = line.substring(12, 16).trim();
          
          atoms.push({ x, y, z, element, atomName, residue });
        }
      } catch (e) {
        // Skip invalid lines
      }
    }
  });

  // Calculate bonds (simple distance-based)
  const bonds: Array<[number, number]> = [];
  const BOND_DISTANCE_THRESHOLD = 2.0; // Angstroms

  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (distance < BOND_DISTANCE_THRESHOLD) {
        bonds.push([i, j]);
      }
    }
  }

  return { atoms, bonds };
}

export function MoleculeViewer({ pdbId, structureUrl }: MoleculeViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const controlsRef = useRef<any>(null);
  const initialCameraDistanceRef = useRef<number>(100);
  const threeRef = useRef<any>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Dynamic import of Three.js
    import('three').then((THREE) => {
      // Store THREE for zoom functions
      threeRef.current = THREE;
      (window as any).THREE = THREE;
      
      // Scene setup with gradient background
      const scene = new THREE.Scene();
      const gradient = new THREE.Color(0x0a0a1a);
      scene.background = gradient;
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(
        60,
        mountRef.current!.clientWidth / mountRef.current!.clientHeight,
        0.1,
        10000
      );
      camera.position.set(0, 0, 100);
      cameraRef.current = camera;

      const renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        alpha: true,
        powerPreference: "high-performance"
      });
      renderer.setSize(mountRef.current!.clientWidth, mountRef.current!.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      mountRef.current!.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Enhanced lighting setup
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
      scene.add(ambientLight);

      const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight1.position.set(50, 50, 50);
      directionalLight1.castShadow = true;
      directionalLight1.shadow.mapSize.width = 2048;
      directionalLight1.shadow.mapSize.height = 2048;
      scene.add(directionalLight1);

      const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
      directionalLight2.position.set(-50, -50, -50);
      scene.add(directionalLight2);

      const pointLight = new THREE.PointLight(0xffffff, 0.6);
      pointLight.position.set(0, 0, 100);
      scene.add(pointLight);

      // Mouse controls for rotation
      let isDragging = false;
      let previousMousePosition = { x: 0, y: 0 };
      const rotationSpeed = 0.005;

      const onMouseDown = (e: MouseEvent) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;
        
        if (sceneRef.current) {
          sceneRef.current.rotation.y += deltaX * rotationSpeed;
          sceneRef.current.rotation.x += deltaY * rotationSpeed;
        }
        
        previousMousePosition = { x: e.clientX, y: e.clientY };
      };

      const onMouseUp = () => {
        isDragging = false;
      };

      renderer.domElement.addEventListener('mousedown', onMouseDown);
      renderer.domElement.addEventListener('mousemove', onMouseMove);
      renderer.domElement.addEventListener('mouseup', onMouseUp);
      renderer.domElement.style.cursor = 'grab';

      // Load PDB structure
      if (structureUrl) {
        fetch(structureUrl)
          .then((response) => response.text())
          .then((pdbData) => {
            const { atoms, bonds } = parsePDB(pdbData);
            
            if (atoms.length === 0) {
              console.warn('No atoms found in PDB file');
              return;
            }

            // Create atom spheres with realistic colors
            const atomGroup = new THREE.Group();
            
            atoms.forEach((atom) => {
              const radius = getAtomRadius(atom.element) * 0.8; // Scale down for better visibility
              const geometry = new THREE.SphereGeometry(radius, 32, 32);
              const color = getAtomColor(atom.element);
              const material = new THREE.MeshPhongMaterial({
                color: color,
                shininess: 100,
                specular: 0x222222,
                emissive: 0x000000,
                transparent: false,
              });
              
              const sphere = new THREE.Mesh(geometry, material);
              sphere.position.set(atom.x, atom.y, atom.z);
              sphere.castShadow = true;
              sphere.receiveShadow = true;
              atomGroup.add(sphere);
            });

            // Create bonds (cylinders between atoms)
            const bondGroup = new THREE.Group();
            const bondGeometry = new THREE.CylinderGeometry(0.15, 0.15, 1, 16);
            const bondMaterial = new THREE.MeshPhongMaterial({
              color: 0xcccccc,
              shininess: 30,
            });

            bonds.forEach(([i, j]) => {
              const atom1 = atoms[i];
              const atom2 = atoms[j];
              
              const midPoint = new THREE.Vector3(
                (atom1.x + atom2.x) / 2,
                (atom1.y + atom2.y) / 2,
                (atom1.z + atom2.z) / 2
              );
              
              const direction = new THREE.Vector3(
                atom2.x - atom1.x,
                atom2.y - atom1.y,
                atom2.z - atom1.z
              );
              
              const length = direction.length();
              direction.normalize();
              
              const cylinder = new THREE.Mesh(bondGeometry, bondMaterial);
              cylinder.scale.y = length;
              cylinder.position.copy(midPoint);
              cylinder.lookAt(atom2.x, atom2.y, atom2.z);
              cylinder.rotateX(Math.PI / 2);
              bondGroup.add(cylinder);
            });

            scene.add(atomGroup);
            scene.add(bondGroup);

            // Center and scale the molecule
            const box = new THREE.Box3();
            box.setFromObject(atomGroup);
            
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 60 / maxDim; // Scale to fit nicely in view

            atomGroup.position.sub(center);
            atomGroup.position.multiplyScalar(scale);
            bondGroup.position.sub(center);
            bondGroup.position.multiplyScalar(scale);
            
            // Scale atom sizes and bond thickness
            atomGroup.children.forEach((child: any) => {
              if (child instanceof THREE.Mesh) {
                child.scale.multiplyScalar(scale);
              }
            });
            
            bondGroup.children.forEach((child: any) => {
              if (child instanceof THREE.Mesh) {
                child.scale.x *= scale;
                child.scale.z *= scale;
              }
            });

            // Adjust camera to view the molecule
            const distance = maxDim * scale * 2;
            initialCameraDistanceRef.current = distance;
            camera.position.set(distance * 0.7, distance * 0.5, distance * 0.7);
            camera.lookAt(0, 0, 0);
          })
          .catch((error) => {
            console.error('Error loading PDB:', error);
            // Show placeholder
            const geometry = new THREE.SphereGeometry(10, 32, 32);
            const material = new THREE.MeshPhongMaterial({ 
              color: 0x444444,
              emissive: 0x111111
            });
            const sphere = new THREE.Mesh(geometry, material);
            scene.add(sphere);
          });
      }

      // Smooth rotation animation
      let autoRotate = true;
      const animate = () => {
        requestAnimationFrame(animate);
        if (sceneRef.current && autoRotate && !isDragging) {
          sceneRef.current.rotation.y += 0.003;
        }
        if (rendererRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };
      animate();

      // Pause auto-rotation on interaction
      renderer.domElement.addEventListener('mouseenter', () => {
        autoRotate = false;
        renderer.domElement.style.cursor = 'grab';
      });
      renderer.domElement.addEventListener('mouseleave', () => {
        autoRotate = true;
        renderer.domElement.style.cursor = 'default';
      });

      // Handle resize
      const handleResize = () => {
        if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;
        cameraRef.current.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      };
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        renderer.domElement.removeEventListener('mousedown', onMouseDown);
        renderer.domElement.removeEventListener('mousemove', onMouseMove);
        renderer.domElement.removeEventListener('mouseup', onMouseUp);
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

  const handleZoomIn = () => {
    const camera = cameraRef.current;
    const THREE = threeRef.current || (window as any).THREE;
    
    if (!camera || !THREE) {
      console.warn('Camera or THREE.js not available');
      return;
    }

    const target = new THREE.Vector3(0, 0, 0);
    const direction = new THREE.Vector3();
    direction.subVectors(camera.position, target).normalize();
    
    // Move camera closer (zoom in)
    camera.position.addScaledVector(direction, -15);
    camera.lookAt(target);
    
    // Force render update
    if (rendererRef.current && sceneRef.current) {
      rendererRef.current.render(sceneRef.current, camera);
    }
  };

  const handleZoomOut = () => {
    const camera = cameraRef.current;
    const THREE = threeRef.current || (window as any).THREE;
    
    if (!camera || !THREE) {
      console.warn('Camera or THREE.js not available');
      return;
    }

    const target = new THREE.Vector3(0, 0, 0);
    const direction = new THREE.Vector3();
    direction.subVectors(camera.position, target).normalize();
    
    // Move camera further (zoom out)
    camera.position.addScaledVector(direction, 15);
    camera.lookAt(target);
    
    // Force render update
    if (rendererRef.current && sceneRef.current) {
      rendererRef.current.render(sceneRef.current, camera);
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full rounded-xl overflow-hidden bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900"
      style={{ minHeight: '400px' }}
    >
      <div
        ref={mountRef}
        className="w-full h-full"
      />
      {/* Zoom Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={handleZoomIn}
          className="glass-card p-2.5 rounded-lg hover:bg-white/20 transition-all duration-200 active:scale-95 shadow-lg backdrop-blur-md bg-white/10 border border-white/20 hover:border-white/30"
          title="Zoom In"
          aria-label="Zoom In"
        >
          <svg
            className="w-5 h-5 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.5}
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
        </button>
        <button
          onClick={handleZoomOut}
          className="glass-card p-2.5 rounded-lg hover:bg-white/20 transition-all duration-200 active:scale-95 shadow-lg backdrop-blur-md bg-white/10 border border-white/20 hover:border-white/30"
          title="Zoom Out"
          aria-label="Zoom Out"
        >
          <svg
            className="w-5 h-5 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.5}
              d="M20 12H4"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
