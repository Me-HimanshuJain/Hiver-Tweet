import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Particles() {
  const count = 100;
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = new THREE.Object3D();

  // Create an array of random positions and speeds for the particles
  const particles = useRef(
    new Array(count).fill(0).map(() => ({
      x: (Math.random() - 0.5) * 20, // Spread across width
      y: (Math.random() - 0.5) * 2,  // Narrow vertical spread
      z: (Math.random() - 0.5) * 2,  // Narrow depth
      speed: 0.05 + Math.random() * 0.05, // Speed of moving right
    }))
  );

  useFrame(() => {
    if (!mesh.current) return;
    
    particles.current.forEach((particle, i) => {
      // Move particle to the right
      particle.x += particle.speed;
      
      // Reset if it goes too far right
      if (particle.x > 10) {
        particle.x = -10;
        particle.y = (Math.random() - 0.5) * 2;
        particle.z = (Math.random() - 0.5) * 2;
      }

      // Update position
      dummy.position.set(particle.x, particle.y, particle.z);
      
      // Scale slightly pulses based on x position
      const scale = 0.5 + Math.sin(particle.x * 2) * 0.2;
      dummy.scale.set(scale, scale, scale);
      
      dummy.updateMatrix();
      mesh.current!.setMatrixAt(i, dummy.matrix);
    });
    
    mesh.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
      <sphereGeometry args={[0.05, 8, 8]} />
      <meshBasicMaterial color="#06B6D4" transparent opacity={0.6} />
    </instancedMesh>
  );
}

function PipelineNodes() {
  const nodes = [
    { label: 'INGEST', x: -8 },
    { label: 'CLASSIFY', x: -3 },
    { label: 'RETRIEVE', x: 2 },
    { label: 'DRAFT', x: 7 },
  ];

  return (
    <group>
      {nodes.map((node, i) => (
        <group key={i} position={[node.x, 0, -1]}>
          <mesh>
            <boxGeometry args={[2, 0.4, 0.1]} />
            <meshBasicMaterial color="#12131a" />
          </mesh>
          <mesh>
            <boxGeometry args={[2.05, 0.45, 0.05]} />
            <meshBasicMaterial color="rgba(255, 255, 255, 0.1)" wireframe />
          </mesh>
        </group>
      ))}
      
      {/* Connecting Line */}
      <mesh position={[-0.5, 0, -1.1]}>
        <boxGeometry args={[16, 0.02, 0.02]} />
        <meshBasicMaterial color="rgba(255, 255, 255, 0.1)" />
      </mesh>
    </group>
  );
}

export default function PipelineVisualization() {
  return (
    <div className="absolute inset-0">
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <color attach="background" args={['#09090B']} />
        <ambientLight intensity={0.5} />
        <Particles />
        <PipelineNodes />
      </Canvas>
    </div>
  );
}
