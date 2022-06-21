//! Imgui Vulkan Renderer.
//! from https://github.com/TheCherno/imgui/blob/1a96ac382aff490a85b5f70403d06adfdcb3b883/backends/imgui_impl_vulkan.cpp

// TODO: multi-viewport
use crate::vulkan;
use ash::vk;

pub type VkResult<T> = Result<T, vk::Result>;

struct FrameSemaphores {
    image_acquired_semaphore: vk::Semaphore,
    render_complete_semaphore: vk::Semaphore,
}

struct MemoryMapGuard<'i, 'a> {
    instance: &'i Instance<'a>,
    ptr: *mut u8,
    memory: vk::DeviceMemory,
    size: usize,
}

use core::borrow::{Borrow, BorrowMut};

impl<T> Borrow<[T]> for MemoryMapGuard<'_, '_> {
    fn borrow(&self) -> &[T] {
        unsafe {
            core::slice::from_raw_parts(self.ptr as *const T, self.size / core::mem::size_of::<T>())
        }
    }
}

impl<T> BorrowMut<[T]> for MemoryMapGuard<'_, '_> {
    fn borrow_mut(&mut self) -> &mut [T] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr as *mut T,
                self.size / core::mem::size_of::<T>(),
            )
        }
    }
}

impl Drop for MemoryMapGuard<'_, '_> {
    fn drop(&mut self) {
        unsafe { self.instance.device.unmap_memory(self.memory) }
    }
}

#[derive(Default)]
struct FrameRenderBuffer {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    handle: vk::Buffer,
}

impl FrameRenderBuffer {
    pub fn needs_reallocation(&self, target_size: vk::DeviceSize) -> bool {
        self.handle == vk::Buffer::null() || self.size < target_size
    }

    /// # Safety
    /// Buffer must be allocated
    pub unsafe fn memory_map<'i, 'a>(
        &self,
        instance: &'i Instance<'a>,
    ) -> VkResult<MemoryMapGuard<'i, 'a>> {
        let ptr =
            instance
                .device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?
                as *mut u8;
        Ok(MemoryMapGuard {
            ptr,
            instance,
            memory: self.memory,
            size: self.size as usize,
        })
    }

    /// # Safety
    /// The buffer must not be in use by the device at reallocation time.
    pub unsafe fn reallocate(
        &mut self,
        instance: &Instance,
        target_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<()> {
        let AllocationInfo {
            buffer,
            buffer_memory,
            buffer_size,
            buffer_memory_alignment: _,
        } = instance.create_or_resize_buffer(
            AllocationInfo {
                buffer: self.handle,
                buffer_size: target_size,
                buffer_memory: self.memory,
                buffer_memory_alignment: 1,
            },
            usage,
        )?;

        self.handle = buffer;
        self.size = buffer_size;
        self.memory = buffer_memory;
        Ok(())
    }

    /// # Safety
    /// The buffer must not be in use by the device at reallocation time.
    pub unsafe fn reallocate_if_needed(
        &mut self,
        instance: &Instance,
        target_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<()> {
        if self.needs_reallocation(target_size) {
            self.reallocate(instance, target_size, usage)?;
        }
        Ok(())
    }
}

pub struct FrameRenderBuffers {
    vertex_buffer: FrameRenderBuffer,
    index_buffer: FrameRenderBuffer,
}

#[derive(Default)]
pub struct WindowRenderBuffers {
    index: u32,
    frame_render_buffers: Option<Box<[FrameRenderBuffers]>>,
}

impl WindowRenderBuffers {
    pub fn current(&self) -> Option<&FrameRenderBuffers> {
        self.frame_render_buffers
            .as_deref()
            .map(|frbs| unsafe { frbs.get_unchecked(self.index as usize) })
    }

    pub fn current_mut(&mut self) -> Option<&mut FrameRenderBuffers> {
        self.frame_render_buffers
            .as_deref_mut()
            .map(|frbs| unsafe { frbs.get_unchecked_mut(self.index as usize) })
    }

    // pub fn advance_index(&mut self) {
    //     self.index = if let Some(frbs) = self.frame_render_buffers.as_deref() {
    //         self.index.wrapping_add(1) % frbs.len() as u32
    //     } else {
    //         0
    //     };
    // }

    pub unsafe fn advance_index_unchecked(&mut self) {
        self.index = self.index.wrapping_add(1)
            % self
                .frame_render_buffers
                .as_deref()
                .unwrap_unchecked()
                .len() as u32;
    }

    pub unsafe fn current_mut_unchecked(&mut self) -> &mut FrameRenderBuffers {
        self.frame_render_buffers
            .as_deref_mut()
            .unwrap_unchecked()
            .get_unchecked_mut(self.index as usize)
    }

    pub fn len(&self) -> usize {
        if let Some(frbs) = self.frame_render_buffers.as_deref() {
            frbs.len()
        } else {
            0
        }
    }
}

/// Helper structure to hold the data needed by one rendering frame
/// Please zero-clear before use!
pub struct Frame {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    backbuffer: vk::Image,
    backbuffer_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
}

impl Default for Frame {
    fn default() -> Self {
        // SAFE: zeroed frame is stil correctly represented.
        unsafe { core::mem::zeroed() }
    }
}

pub struct Window {
    width: u32,
    height: u32,
    swapchain: vk::SwapchainKHR,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    render_pass: vk::RenderPass,
    /// The window pipeline may use a different renderPass than the one passed in Instance
    pipeline: vk::Pipeline,
    clear_enable: bool,
    clear_value: vk::ClearValue,
    /// Current frame beind rendered to (0 <= frame_index < frame_in_flight_count)
    frame_index: u32,
    /// Number of simultaneous in-flight frames (returned by `vkGetSwapchinImagesKHR`, usualy
    /// derived from [`min_image_count`](Instance::min_image_count)
    image_count: u32,
    /// Current set of swapchain wait semaphores we're using (needs to be distinct from per frame
    /// data)
    semaphore_index: u32,
    frames: Vec<Frame>,
    frame_semaphores: Vec<Frame>,
}

/// Vulkan renderer state
pub struct Renderer {
    render_pass: vk::RenderPass,
    buffer_memory_alignment: vk::DeviceSize,
    pipeline_create_flags: vk::PipelineCreateFlags,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    subpass: u32,
    shader_module_vert: vk::ShaderModule,
    shader_module_frag: vk::ShaderModule,

    // Font data
    font_sampler: vk::Sampler,
    font_memory: vk::DeviceMemory,
    font_image: vk::Image,
    font_view: vk::ImageView,
    font_descriptor_set: vk::DescriptorSet,
    upload_buffer_memory: vk::DeviceMemory,
    upload_buffer: vk::Buffer,

    // Render buffers for main window
    main_window_render_buffers: WindowRenderBuffers,
    viewport_data: ViewportData,
}

// For multi-viewport support:
struct ViewportData {
    window_owned: bool,
    window: Window,
    render_buffers: WindowRenderBuffers,
}

impl Renderer {
    // FIXME: experimental.
    pub fn add_texture(
        &self,
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        sampler: vk::Sampler,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) -> VkResult<vk::DescriptorSet> {
        // Create descriptor_set
        let descriptor_set = {
            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.descriptor_set_layout,
                ..Default::default()
            };

            // SAFE: alloc info is valid.
            // SAFE: we only allocate 1 descriptor set
            unsafe {
                device
                    .allocate_descriptor_sets(&alloc_info)?
                    .pop()
                    .unwrap_unchecked()
            }
        };

        // update the descriptor set
        {
            let desc_image = [vk::DescriptorImageInfo {
                sampler,
                image_view,
                image_layout,
            }];

            let write_desc = [vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                descriptor_count: desc_image.len() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: desc_image.as_ptr(), // SAFE: length was set to desc_image.len()
                ..Default::default()
            }];

            // SAFE: data is correct.
            unsafe {
                device.update_descriptor_sets(&write_desc, &[]);
            }
        }

        Ok(descriptor_set)
    }

    pub fn render_draw_data(
        &mut self,
        instance: &Instance,
        draw_data: &mut imgui::DrawData,
        command_buffer: vk::CommandBuffer,
        pipeline: Option<vk::Pipeline>,
    ) -> VkResult<()> {
        // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if fb_width <= 0.0 || fb_height <= 0.0 {
            return Ok(());
        }

        let pipeline = pipeline.unwrap_or(self.pipeline);

        // Allocate array to store enough vetrex/index buffers. Each unique viewport gets its own
        // storage.
        {
            let viewport_renderer_data = &mut self.viewport_data;
            let wrb = &mut viewport_renderer_data.render_buffers;
            if wrb.frame_render_buffers.is_none() {
                wrb.index = 0;
                wrb.frame_render_buffers = Some(unsafe {
                    Box::new_zeroed_slice(instance.image_count as usize).assume_init()
                });
            }
            // SAFE: we just assigned frame render buffers if they were null.
            unsafe {
                wrb.advance_index_unchecked();
            }
        }

        {
            // SAFE: same as just above
            let rb = unsafe { self.viewport_data.render_buffers.current_mut_unchecked() };

            if draw_data.total_vtx_count > 0 {
                // Create or resize the vertei/index buffers
                let vertex_size =
                    draw_data.total_vtx_count as usize * core::mem::size_of::<imgui::DrawVert>();
                let index_size =
                    draw_data.total_idx_count as usize * core::mem::size_of::<imgui::DrawVert>();

                unsafe {
                    rb.vertex_buffer.reallocate_if_needed(
                        instance,
                        vertex_size as vk::DeviceSize,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    )?;
                    rb.index_buffer.reallocate_if_needed(
                        instance,
                        index_size as vk::DeviceSize,
                        vk::BufferUsageFlags::INDEX_BUFFER,
                    )?;
                }

                // Upload vertex/index data into a single contiguous GPU buffer
                let mut vtx_guard = unsafe { rb.vertex_buffer.memory_map(instance) }?;
                let mut idx_guard = unsafe { rb.index_buffer.memory_map(instance) }?;
                let vtx_dst: &mut [imgui::DrawVert] = vtx_guard.borrow_mut();
                let idx_dst: &mut [imgui::DrawIdx] = idx_guard.borrow_mut();

                let mut vertex_index = 0;
                let mut index_index = 0;
                for draw_list in draw_data.draw_lists() {
                    let vtx_range = vertex_index..vertex_index + draw_list.vtx_buffer().len();
                    let idx_range = index_index..index_index + draw_list.idx_buffer().len();

                    vtx_dst[vtx_range].copy_from_slice(draw_list.vtx_buffer());
                    idx_dst[idx_range].copy_from_slice(draw_list.idx_buffer());

                    vertex_index += draw_list.vtx_buffer().len();
                    index_index += draw_list.idx_buffer().len();
                }

                let memory_ranges = [
                    vk::MappedMemoryRange {
                        memory: rb.vertex_buffer.memory,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    },
                    vk::MappedMemoryRange {
                        memory: rb.index_buffer.memory,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    },
                ];

                unsafe { instance.device.flush_mapped_memory_ranges(&memory_ranges) }?;
            }

            // Set the desired Vulkan state
            setup_render_state(
                draw_data,
                &instance.device,
                pipeline,
                self.pipeline_layout,
                command_buffer,
                rb,
                fb_width,
                fb_height,
            );

            // Will project scissor/clipping rectangles into framebuffer space
            let clip_off = draw_data.display_pos; // (0,0)  unless using multi-viewports
            let clip_scale = draw_data.framebuffer_scale; // (1,1) unless using retina display
                                                          // which are often (2,2)

            // Render command lists
            // (Because we merged all buffers into a single one, we maintain our own offset
            // into them)
            let mut global_vtx_offset = 0;
            let mut global_idx_offset = 0;
            for draw_list in draw_data.draw_lists() {
                for cmd in draw_list.commands() {
                    match cmd {
                        imgui::DrawCmd::Elements { count, cmd_params } => {
                            // Project scissor/clipping rectangles into framebuffer space
                            // Clamp to iewporta s vkCmdSetScissor() won't accept values that are
                            // off bounds
                            let clip_min = [
                                ((cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0]).max(0.0),
                                ((cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1]).max(0.0),
                            ];
                            let clip_max = [
                                ((cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0])
                                    .min(fb_width),
                                ((cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1])
                                    .min(fb_height),
                            ];

                            if clip_max[0] <= clip_min[0] || clip_max[1] <= clip_min[1] {
                                continue; // skip if viewport is null
                            }

                            let scissor = vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: clip_min[0] as i32,
                                    y: clip_min[1] as i32,
                                },
                                extent: vk::Extent2D {
                                    width: (clip_max[0] - clip_min[0]) as u32,
                                    height: (clip_max[1] - clip_min[1]) as u32,
                                },
                            };

                            unsafe {
                                instance
                                    .device
                                    .cmd_set_scissor(command_buffer, 0, &[scissor])
                            };
                            use vk::Handle;

                            // Bind dsecriptor set with fount or user texture
                            let desc_set = if core::mem::size_of::<imgui::sys::ImTextureID>()
                                < core::mem::size_of::<u64>()
                            {
                                // We don't support texture switches if ImTextureId hasn't been
                                // redefined to be 64-bit. Do a flaky check that other textures
                                // haven't been used.
                                assert_eq!(
                                    cmd_params.texture_id.id() as u64,
                                    self.font_descriptor_set.as_raw(),
                                );
                                self.font_descriptor_set
                            } else {
                                // NOTE: imgui impl makes a cast, so I guess I can do it too...
                                vk::DescriptorSet::from_raw(cmd_params.texture_id.id() as u64)
                            };

                            unsafe {
                                instance.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    self.pipeline_layout,
                                    0,
                                    &[desc_set],
                                    &[],
                                );
                            }

                            // draw
                            unsafe {
                                instance.device.cmd_draw_indexed(
                                    command_buffer,
                                    count as u32,
                                    1,
                                    (cmd_params.idx_offset + global_idx_offset) as u32,
                                    (cmd_params.vtx_offset + global_vtx_offset) as i32,
                                    0,
                                );
                            }
                        }
                        imgui::DrawCmd::ResetRenderState => {
                            setup_render_state(
                                draw_data,
                                &instance.device,
                                pipeline,
                                self.pipeline_layout,
                                command_buffer,
                                rb,
                                fb_width,
                                fb_height,
                            );
                        }
                        imgui::DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                            callback(draw_list.raw(), raw_cmd)
                        },
                    }
                }
                use imgui::internal::RawWrapper;
                unsafe {
                    global_idx_offset += draw_list.raw().IdxBuffer.Size as usize;
                    global_vtx_offset += draw_list.raw().VtxBuffer.Size as usize;
                }
            }
        }

        // Note: at this point both vkCmdSetViewport() and vkCmdSetScissor() have been called.
        // Our last values will leak into user/application rendering IF:
        // - Your app uses a pipeline with VK_DYNAMIC_STATE_VIEWPORT or VK_DYNAMIC_STATE_SCISSOR dynamic state
        // - And you forgot to call vkCmdSetViewport() and vkCmdSetScissor() yourself to explicitely set that state.
        // If you use VK_DYNAMIC_STATE_VIEWPORT or VK_DYNAMIC_STATE_SCISSOR you are responsible for setting the values before rendering.
        // In theory we should aim to backup/restore those values but I am not sure this is possible.
        // We perform a call to vkCmdSetScissor() to set back a full viewport which is likely to fix things for 99% users but technically this is not perfect. (See github #4644)
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: fb_width as u32,
                height: fb_height as u32,
            },
        };
        unsafe {
            instance
                .device
                .cmd_set_scissor(command_buffer, 0, &[scissor])
        }

        Ok(())
    }
}

fn setup_render_state(
    draw_data: &imgui::DrawData,
    device: &ash::Device,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    command_buffer: vk::CommandBuffer,
    rb: &mut FrameRenderBuffers,
    fb_width: f32,
    fb_height: f32,
) {
    // Bind pipeline
    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
    }

    // Bind vertex and index buffer
    if draw_data.total_vtx_count > 0 {
        let vertex_buffers = [rb.vertex_buffer.handle];
        let vertex_offset = [0 as vk::DeviceSize];

        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offset);
            device.cmd_bind_index_buffer(
                command_buffer,
                rb.index_buffer.handle,
                0,
                if core::mem::size_of::<imgui::DrawIdx>() == 2 {
                    vk::IndexType::UINT16
                } else {
                    vk::IndexType::UINT32
                },
            );
        }
    }

    // Setup viewport
    unsafe {
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: 0.0f32,
                y: 0.0f32,
                width: fb_width,
                height: fb_height,
                min_depth: 0.0f32,
                max_depth: 1.0f32,
                ..Default::default()
            }],
        )
    }

    // Setup scale and translation:
    // Our visible imgui space lies from draw_data->DisplayPps (top left) to
    // draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos
    // is (0,0) for single viewport apps.
    let scale = [
        2.0f32 / draw_data.display_size[0],
        2.0f32 / draw_data.display_size[1],
    ];
    let translate = [
        -1.0f32 - draw_data.display_size[0] * scale[0],
        -1.0f32 - draw_data.display_size[1] * scale[1],
    ];

    unsafe {
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            slice_as_bytes(&scale),
        );
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            (scale.len() * core::mem::size_of::<f32>()) as u32,
            slice_as_bytes(&translate),
        )
    }
}

pub fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFE:
    //  1. T is POD.
    //  2. length is multiplied by sizeof(T)
    unsafe {
        core::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * core::mem::size_of::<T>(),
        )
    }
}
/// ImGui Vulkan Instance.
pub struct Instance<'alloc> {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub pipeline_cache: vk::PipelineCache,
    pub descriptor_pool: vk::DescriptorPool,
    pub subpass: u32,
    /// Must be >= 2
    pub min_image_count: u32,
    /// Must be >= min_image_count
    pub image_count: u32,
    pub msaa_samples: vk::SampleCountFlags,
    pub allocation_callbacks: Option<&'alloc vk::AllocationCallbacks>,
}

pub struct AllocationInfo {
    pub buffer: vk::Buffer,
    pub buffer_memory: vk::DeviceMemory,
    pub buffer_memory_alignment: vk::DeviceSize,
    pub buffer_size: vk::DeviceSize,
}
impl Instance<'_> {
    #[inline]
    pub fn memory_type_index(&self, properties: vk::MemoryPropertyFlags, type_bits: u32) -> u32 {
        vulkan::memory_type_index(&self.instance, self.physical_device, properties, type_bits)
    }

    /// Create or resize a buffer.
    /// # Safety
    /// Requires:
    ///     - the buffer to (re)allocate must not be in use by the device
    ///     - size must be >= 1
    pub unsafe fn create_or_resize_buffer(
        &self,
        AllocationInfo {
            mut buffer,
            mut buffer_memory,
            mut buffer_memory_alignment,
            buffer_size,
        }: AllocationInfo,
        usage: vk::BufferUsageFlags,
    ) -> vulkan::Result<AllocationInfo> {
        if buffer != vk::Buffer::null() {
            // SAFE: premise
            unsafe {
                self.device
                    .destroy_buffer(buffer, self.allocation_callbacks);
            }
        }
        if buffer_memory != vk::DeviceMemory::null() {
            // SAFE: premise
            unsafe {
                self.device
                    .free_memory(buffer_memory, self.allocation_callbacks);
            }
        }

        let vertex_buffer_size_aligned =
            ((buffer_size - 1) / buffer_memory_alignment) * buffer_memory_alignment;

        let buffer_create_info = vk::BufferCreateInfo {
            size: vertex_buffer_size_aligned,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        // SAFE: create info struct and allocation callbacks are valid.
        buffer = unsafe {
            self.device
                .create_buffer(&buffer_create_info, self.allocation_callbacks)
        }?;

        // SAFE: buffer is valid.
        let memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        buffer_memory_alignment = buffer_memory_alignment.max(memory_requirements.alignment);

        let memory_alloc_info = vk::MemoryAllocateInfo {
            allocation_size: memory_requirements.size,
            memory_type_index: self.memory_type_index(
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                memory_requirements.memory_type_bits,
            ),
            ..Default::default()
        };

        buffer_memory = unsafe {
            self.device
                .allocate_memory(&memory_alloc_info, self.allocation_callbacks)
        }?;

        self.device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok(AllocationInfo {
            buffer,
            buffer_memory,
            buffer_memory_alignment,
            buffer_size: memory_requirements.size,
        })
    }
}
