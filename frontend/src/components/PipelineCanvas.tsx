import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const STAGES = [
  { x: -7, name: 'INGEST', color: '#00E5FF' },
  { x: -3.5, name: 'CLASSIFY', color: '#00E5FF' },
  { x: 0, name: 'GROUNDING', color: '#38BDF8' },
  { x: 3.5, name: 'GENERATION', color: '#FFAA00' },
  { x: 7, name: 'ARBITRATION', color: '#10B981' }
];

const Node = ({ x, color, index }: { x: number; color: string; index: number }) => {
  const meshRef = useRef<THREE.Mesh>(null!);
  const ringRef = useRef<THREE.Mesh>(null!);
  const speed = 0.015 * (index % 2 === 0 ? 1 : -1);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += speed;
      meshRef.current.rotation.x += speed * 0.5;
    }
    if (ringRef.current) {
      ringRef.current.rotation.z += 0.01;
    }
  });

  return (
    <group position={[x, 0, 0]}>
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[0.75, 1]} />
        <meshPhongMaterial color={color} wireframe transparent opacity={0.85} />
        <mesh>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.9} />
        </mesh>
      </mesh>
      <mesh ref={ringRef} rotation={[Math.PI / 2 + index * 0.4, 0, 0]}>
        <torusGeometry args={[1.1, 0.02, 8, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.4} />
      </mesh>
    </group>
  );
};

const Particles = () => {
  const particleCount = 280;
  const positions = useMemo(() => new Float32Array(particleCount * 3), []);
  const colors = useMemo(() => new Float32Array(particleCount * 3), []);
  const speeds = useMemo(() => new Float32Array(particleCount), []);
  
  const pointsRef = useRef<THREE.Points>(null!);

  useMemo(() => {
    const colA = new THREE.Color('#00E5FF');
    const colB = new THREE.Color('#FFAA00');
    const colC = new THREE.Color('#10B981');

    for (let i = 0; i < particleCount; i++) {
      const px = (Math.random() - 0.5) * 16;
      const py = (Math.random() - 0.5) * 1.8;
      const pz = (Math.random() - 0.5) * 1.8;
      positions[i * 3] = px;
      positions[i * 3 + 1] = py;
      positions[i * 3 + 2] = pz;

      const t = (px + 8) / 16;
      const mixed = t < 0.6 ? colA.clone().lerp(colB, t / 0.6) : colB.clone().lerp(colC, (t - 0.6) / 0.4);
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;

      speeds[i] = 0.04 + Math.random() * 0.05;
    }
  }, [positions, colors, speeds]);

  useFrame(() => {
    if (pointsRef.current) {
      const posAttr = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < particleCount; i++) {
        let px = posAttr.getX(i);
        px += speeds[i];
        if (px > 8.5) px = -8.5;
        posAttr.setX(i, px);
      }
      posAttr.needsUpdate = true;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.07} vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} />
    </points>
  );
};

const Connections = () => {
  return (
    <>
      {STAGES.slice(0, -1).map((st, i) => {
        const next = STAGES[i + 1];
        const curve = new THREE.LineCurve3(new THREE.Vector3(st.x, 0, 0), new THREE.Vector3(next.x, 0, 0));
        return (
          <mesh key={`conn-${i}`}>
            <tubeGeometry args={[curve, 20, 0.04, 6, false]} />
            <meshBasicMaterial color="#00E5FF" transparent opacity={0.25} />
          </mesh>
        );
      })}
    </>
  );
};

const Scene = () => {
  const groupRef = useRef<THREE.Group>(null!);
  const [mouse, setMouse] = React.useState({ x: 0, y: 0 });

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMouse({
        x: (e.clientX / window.innerWidth - 0.5) * 0.3,
        y: (e.clientY / window.innerHeight - 0.5) * 0.2
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += (mouse.x - groupRef.current.rotation.y) * 0.04;
      groupRef.current.rotation.x += (-mouse.y - groupRef.current.rotation.x) * 0.04;
    }
  });

  return (
    <group ref={groupRef}>
      {STAGES.map((st, i) => (
        <Node key={st.name} x={st.x} color={st.color} index={i} />
      ))}
      <Particles />
      <Connections />
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 10, 7]} color="#00e5ff" intensity={1.2} />
    </group>
  );
};

export const PipelineCanvas = () => {
  return (
    <section className="relative w-full rounded-2xl bg-glass-fill/60 backdrop-blur-xl border border-cyan/20 p-5 overflow-hidden shadow-2xl">
      {/* Top HUD Header Overlay */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-3 pb-3 border-b border-cyan/10 z-10 relative">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-surface-container-high/90 border border-cyan/20 px-3 py-1 rounded-md">
            <span className="material-symbols-outlined text-primary-container text-[18px]">polyline</span>
            <span className="font-meta-sm text-meta-sm font-semibold tracking-wider text-primary">NEURAL PIPELINE TOPOLOGY // REAL-TIME EVENT BUS</span>
          </div>
          <span className="font-meta-sm text-meta-sm text-on-surface-variant hidden md:inline">Sync Frequency: 100Hz</span>
        </div>
        <div className="flex items-center space-x-2">
          <button className="px-3 py-1.5 rounded-lg bg-surface-container-high hover:bg-surface-bright border border-cyan/20 text-on-surface-variant hover:text-starlight-white font-meta-sm text-meta-sm flex items-center space-x-1.5 active:-translate-y-px transition-all">
            <span className="material-symbols-outlined text-[16px]">pause_circle</span>
            <span>Pause Stream</span>
          </button>
          <button className="px-3 py-1.5 rounded-lg bg-surface-container-high hover:bg-surface-bright border border-cyan/20 text-on-surface-variant hover:text-starlight-white font-meta-sm text-meta-sm flex items-center space-x-1.5 active:-translate-y-px transition-all">
            <span className="material-symbols-outlined text-[16px]">sync</span>
            <span>Re-index RAG</span>
          </button>
          <button className="px-3.5 py-1.5 rounded-lg bg-secondary-container text-on-secondary-container hover:bg-secondary-fixed font-meta-sm text-meta-sm font-bold flex items-center space-x-1.5 active:-translate-y-px transition-all">
            <span className="material-symbols-outlined text-[16px]">tune</span>
            <span>Config Thresholds</span>
          </button>
        </div>
      </div>
      
      {/* 3D Animation Custom Element */}
      <div className="relative w-full h-72 rounded-xl overflow-hidden bg-surface-container-lowest/80 border border-cyan/10">
        <Canvas camera={{ position: [0, 0.5, 8.5], fov: 60 }}>
          <Scene />
        </Canvas>
        
        {/* Subtle HUD overlay grid lines */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_0%,rgba(5,20,36,0.6)_100%)]"></div>
        <div className="absolute bottom-3 left-4 pointer-events-none font-meta-sm text-[11px] text-on-surface-variant/70 flex items-center space-x-4">
          <span>LATENCY BUDGET: 250ms</span>
          <span>VECTOR SHARDS: 64/64 ACTIVE</span>
          <span>MEMORY BUFFER: 2.1GB / 8.0GB</span>
        </div>
      </div>

      {/* 5-Stage Telemetry Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
        {/* Stage 1 */}
        <div className="p-3 rounded-lg bg-surface-container-low/70 border border-cyan/20 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="font-meta-sm text-[11px] text-on-surface-variant">STAGE 01</span>
            <span className="w-2 h-2 rounded-full bg-primary-container"></span>
          </div>
          <span className="font-headline-md text-sm font-bold text-starlight-white">Ingestion</span>
          <div className="flex items-center justify-between mt-2 pt-1 border-t border-cyan/10 font-meta-sm text-[11px]">
            <span className="text-on-surface-variant">Kafka Direct</span>
            <span className="text-primary-fixed-dim">0.8ms</span>
          </div>
        </div>
        {/* Stage 2 */}
        <div className="p-3 rounded-lg bg-surface-container-low/70 border border-cyan/20 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="font-meta-sm text-[11px] text-on-surface-variant">STAGE 02</span>
            <span className="w-2 h-2 rounded-full bg-primary-container"></span>
          </div>
          <span className="font-headline-md text-sm font-bold text-starlight-white">Intent Classifier</span>
          <div className="flex items-center justify-between mt-2 pt-1 border-t border-cyan/10 font-meta-sm text-[11px]">
            <span className="text-on-surface-variant">RoBERTa-Hiver</span>
            <span className="text-primary-fixed-dim">12.4ms</span>
          </div>
        </div>
        {/* Stage 3 */}
        <div className="p-3 rounded-lg bg-surface-container-low/70 border border-cyan/20 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="font-meta-sm text-[11px] text-on-surface-variant">STAGE 03</span>
            <span className="w-2 h-2 rounded-full bg-secondary-container"></span>
          </div>
          <span className="font-headline-md text-sm font-bold text-starlight-white">Vector RAG</span>
          <div className="flex items-center justify-between mt-2 pt-1 border-t border-cyan/10 font-meta-sm text-[11px]">
            <span className="text-on-surface-variant">Milvus Qdrant</span>
            <span className="text-secondary-fixed">41.2ms</span>
          </div>
        </div>
        {/* Stage 4 */}
        <div className="p-3 rounded-lg bg-surface-container-low/70 border border-cyan/20 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-1">
            <span className="font-meta-sm text-[11px] text-on-surface-variant">STAGE 04</span>
            <span className="w-2 h-2 rounded-full bg-primary-container"></span>
          </div>
          <span className="font-headline-md text-sm font-bold text-starlight-white">Synthesizer</span>
          <div className="flex items-center justify-between mt-2 pt-1 border-t border-cyan/10 font-meta-sm text-[11px]">
            <span className="text-on-surface-variant">Nemotron-120B</span>
            <span className="text-primary-fixed-dim">78.1ms</span>
          </div>
        </div>
        {/* Stage 5 */}
        <div className="p-3 rounded-lg bg-surface-container-low/70 border border-cyan/20 flex flex-col justify-between col-span-2 md:col-span-1">
          <div className="flex items-center justify-between mb-1">
            <span className="font-meta-sm text-[11px] text-on-surface-variant">STAGE 05</span>
            <span className="w-2 h-2 rounded-full bg-primary-container"></span>
          </div>
          <span className="font-headline-md text-sm font-bold text-starlight-white">Arbitration</span>
          <div className="flex items-center justify-between mt-2 pt-1 border-t border-cyan/10 font-meta-sm text-[11px]">
            <span className="text-on-surface-variant">Guardrail-v2</span>
            <span className="text-primary-fixed-dim">9.5ms</span>
          </div>
        </div>
      </div>
    </section>
  );
};
