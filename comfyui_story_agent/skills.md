# Agent Skills

These skills define the core competencies of the specialized agents within the video generation pipeline.

## Social Media Marketing Expert
- **Audience Analysis & Trend Spotting**: Identifying target demographics and aligning visual styles with current platform algorithms and trends.
- **Engagement Optimization**: Structuring video pacing to include strong hooks in the first 3 seconds to maximize viewer retention and reduce bounce rates.
- **Value Proposition Display**: Designing shots that clearly communicate the product's functionality, benefits, and emotional value quickly and intuitively.
- **Call-To-Action (CTA) Integration**: Organically working clear actions into the video flow.
- **Platform-Specific Formats**: Adapting content aspect ratios, metadata, and visual framing specifically tailored for platforms like TikTok, Instagram Reels, and YouTube Shorts.

## Videography and Editing Professional
- **Cinematic Composition**: Applying professional framing rules (Rule of Thirds, leading lines, golden ratio, symmetry) to create visually striking and balanced imagery.
- **Advanced Lighting Techniques**: Utilizing three-point lighting, dramatic shadows, practical lights, and cinematic color grading to evoke mood and emphasize high-end product quality.
- **Dynamic Camera Movement**: Planning motivated camera motions (pans, tilts, dolly zooms, tracking shots) that enhance the narrative and product functionality without distracting from the subject.
- **Visual Continuity & Flow**: Ensuring seamless flow between cuts via match-on-action, spatial consistency, and logical visual progression across the storyboard.
- **Depth of Field & Focal Control**: Mastering background blur (bokeh) and focus pulls to direct viewer attention specifically to the most important elements of the frame.
- **Pacing and Rhythm**: Editing shots with a cadence that matches the energy of the concept and keeps viewers visually engaged.

## UI and Text Rendering (2026 Standards)
- **Native Video Models**: For projects requiring native text generation, prioritize workflows integrating **Kling 3.0 (Omni)** for its industry-leading "native-level precision" and physical realism that prevents text from melting during camera movements.
- **Image-to-Video Base Generation**: When using I2V workflows, generate base anchor images using **Ideogram** to ensure highly accurate, legible typography before animation begins.
- **The Bulletproof Hybrid Workaround**: 
  1. Generate the base scene using a top-tier model without expecting accurate UI generation.
  2. Use a "Motion Brush" or masking node to freeze the screen area (setting motion parameters to zero).
  3. Overlay a high-quality, real graphic of the app interface onto the screen in post-production.
  4. Motion track the overlaid graphic to follow camera and subject movements naturally.

## Marketing Models & Workflows (2026 Standards)
- **Top Base Models**: Utilize **Z-Image Turbo (ZiT)** for high-speed, realistic scene generation. Use **Flux 1 Dev / Flux 2** for high-end commercial photo-shoot aesthetics and structural consistency.
- **Product & UI Compositing**: For strict prompt following and stitching products into locations, use **Qwen-Image-Edit** or the **Flux Kontext** workflow (e.g., `i2i_by_flux_kontext_pro.json`) to perfectly blend base photos with desired scenes.
- **Essential Nodes**:
  - **IC-Light**: Critical for relighting newly generated subjects or backgrounds to match the original product's light source.
  - **IP-Adapter**: Mandatory for style and character transfer to maintain identical brand aesthetics across multiple campaigns.
- **LoRAs**: 
  - **Lincoln Studio Video Reasoning LoRA**: A critical LTX 2.3 LoRA trained on 1M+ videos for advanced physical reasoning, object trajectories, collisions, and gravity. Greatly improves motion dynamics and reduces hallucinations, especially in image-to-video (I2V) workflows.
  - **Aesthetic & Detail LoRAs**: Incorporate Aesthetic & Color Grading LoRAs (like Amateur Photo Look) and Detail Enhancement LoRAs (DMD, NoobAI) to achieve commercial texture quality efficiently.
