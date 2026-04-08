import omni.replicator.core as rep

with rep.new_layer():
    # 1. Setup Camera & Render Product
    camera = rep.create.camera(position=(0, 0, 150), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, (1024, 1024))
    
    # 2. Create the Table
    table = rep.create.cube(semantics=[('class', 'table')], position=(0, 0, -10), scale=(100, 100, 1))

    # 3. Domain Randomization: EMNIST Targets
    def randomize_emnist_targets():
        # Spawn planes to act as canvases for EMNIST images
        targets = rep.create.plane(count=5, semantics=[('class', 'emnist_target')])
        with targets:
            rep.modify.pose(
                # Randomize positions on the table surface
                position=rep.distribution.uniform((-40, -40, 0), (40, 40, 0)),
                # Randomize Skew/Scale
                scale=rep.distribution.uniform((1, 1, 1), (3, 5, 1)),
                rotation=rep.distribution.uniform((0, 0, -180), (0, 0, 180))
            )
            # You will map the EMNIST images as materials/textures here
        return targets

    # 4. Domain Randomization: Insertion/Occluders
    def insert_occluders():
        # Spawn random physical blocks to occlude the targets
        distractors = rep.create.cube(count=8, semantics=[('class', 'distractor')])
        with distractors:
            rep.modify.pose(
                # Ensure they spawn above the targets (Z > 0) to cause occlusion
                position=rep.distribution.uniform((-40, -40, 2), (40, 40, 10)),
                scale=rep.distribution.uniform((2, 2, 2), (8, 8, 8))
            )
        return distractors

    # Register the randomizers
    rep.randomizer.register(randomize_emnist_targets)
    rep.randomizer.register(insert_occluders)

    # 5. Trigger generation for 1000 frames
    with rep.trigger.on_frame(num_frames=1000):
        rep.randomizer.randomize_emnist_targets()
        rep.randomizer.insert_occluders()

    # 6. Record the Dataset
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="emnist_active_inference_dataset", 
        rgb=True, 
        bounding_box_2d_tight=True, 
        semantic_segmentation=True
    )
    writer.attach([render_product])
