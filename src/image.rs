// use crate::imgui;
// use crate::vulkan;
// use crate::vulkan::AllocationInfo;
// use ash::vk;
// use stb_image as stb;
// use std::io;
//
// #[derive(Debug, Clone, Copy)]
// pub enum Format {
//     None = 0,
//     RGBA = 1,
//     RGBA32F = 2,
// }
//
// impl Format {
//     fn bytes_per_pixel(self) -> u64 {
//         match self {
//             Self::None => 0,
//             Self::RGBA => 4,
//             Self::RGBA32F => 16,
//         }
//     }
//
//     fn as_vulkan_format(self) -> vk::Format {
//         match self {
//             Self::None => vk::Format::UNDEFINED,
//             Self::RGBA => vk::Format::R8G8B8A8_UNORM,
//             Self::RGBA32F => vk::Format::R32G32B32A32_SFLOAT,
//         }
//     }
// }
//
// impl Default for Format {
//     fn default() -> Self {
//         Self::None
//     }
// }
//
// pub struct Image {
//     width: u32,
//     height: u32,
//     descriptor_set: vk::DescriptorSet,
//     image: vk::Image,
//     image_view: vk::ImageView,
//     memory: vk::DeviceMemory,
//     sampler: vk::Sampler,
//     format: Format,
//     staging_buffer: vk::Buffer,
//     staging_buffer_memory: vk::DeviceMemory,
//     aligned_size: vk::DeviceSize,
// }
//
// impl Image {
//     pub fn new(width: usize, height: usize, format: Format, data: &[u8]) -> Self {
//         todo!()
//     }
//
//     fn allocate_memory(
//         &mut self,
//         instance: &vulkan::Instance,
//         imgui_renderer: &imgui::Renderer,
//     ) -> vulkan::Result<()> {
//         (self.image, self.memory) = {
//             let create_info = vk::ImageCreateInfo::builder()
//                 .image_type(vk::ImageType::TYPE_2D)
//                 .format(self.format.as_vulkan_format())
//                 .extent(vk::Extent3D {
//                     width: self.width,
//                     height: self.height,
//                     depth: 1,
//                 })
//                 .mip_levels(1)
//                 .array_layers(1)
//                 .samples(vk::SampleCountFlags::TYPE_1)
//                 .tiling(vk::ImageTiling::OPTIMAL)
//                 .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
//                 .sharing_mode(vk::SharingMode::EXCLUSIVE)
//                 .initial_layout(vk::ImageLayout::UNDEFINED)
//                 .build();
//
//             let image = unsafe { instance.device.create_image(&create_info, None) }?;
//
//             let mem_requirements = unsafe {
//                 instance
//                     .device
//                     .get_buffer_memory_requirements(self.staging_buffer)
//             };
//
//             let allocate_info = vk::MemoryAllocateInfo::builder()
//                 .allocation_size(self.width as u64 * self.height as u64)
//                 .memory_type_index(instance.memory_type_index(
//                     vk::MemoryPropertyFlags::DEVICE_LOCAL,
//                     mem_requirements.memory_type_bits,
//                 ))
//                 .build();
//
//             let memory = unsafe { instance.device.allocate_memory(&allocate_info, None) }?;
//
//             unsafe {
//                 instance.device.bind_image_memory(image, memory, 0)?;
//             }
//
//             (image, memory)
//         };
//
//         // create the image view
//         self.image_view = {
//             let create_info = vk::ImageViewCreateInfo::builder()
//                 .image(self.image)
//                 .view_type(vk::ImageViewType::TYPE_2D)
//                 .format(self.format.as_vulkan_format())
//                 .subresource_range(
//                     vk::ImageSubresourceRange::builder()
//                         .aspect_mask(vk::ImageAspectFlags::COLOR)
//                         .level_count(1)
//                         .layer_count(1)
//                         .build(),
//                 )
//                 .build();
//             unsafe { instance.device.create_image_view(&create_info, None) }?
//         };
//
//         // create sampler
//         self.sampler = {
//             let create_info = vk::SamplerCreateInfo::builder()
//                 .mag_filter(vk::Filter::LINEAR)
//                 .min_filter(vk::Filter::LINEAR)
//                 .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
//                 .address_mode_u(vk::SamplerAddressMode::REPEAT)
//                 .address_mode_v(vk::SamplerAddressMode::REPEAT)
//                 .address_mode_w(vk::SamplerAddressMode::REPEAT)
//                 .min_lod(-1000.0)
//                 .max_lod(1000.0)
//                 .max_anisotropy(1.0)
//                 .build();
//             unsafe { instance.device.create_sampler(&create_info, None) }?
//         };
//
//         self.descriptor_set = imgui_renderer.add_texture(
//             instance,
//             self.sampler,
//             self.image_view,
//             vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
//         )?;
//
//         Ok(())
//     }
//
//     /// Sets the new data.
//     /// # Safety
//     /// 1- The image buffer must not be in use by the device
//     /// 2- The given `data` must be the same size as: width * height * format.bytes_per_pixel()
//     pub unsafe fn set_data(
//         &mut self,
//         instance: &vulkan::Instance,
//         data: &[u8],
//         device: &ash::Device,
//     ) -> vulkan::Result<()> {
//         let upload_size = self.width as u64 * self.height as u64 * self.format.bytes_per_pixel();
//         debug_assert_eq!(
//             data.len(),
//             upload_size as usize,
//             "Data must be exact image size!"
//         );
//
//         if self.staging_buffer == vk::Buffer::null() {
//             // create the upload buffer
//             // SAFE: premise 1.
//             let AllocationInfo {
//                 buffer,
//                 buffer_memory,
//                 buffer_memory_alignment,
//                 buffer_size,
//             } = unsafe {
//                 instance.create_or_resize_buffer(
//                     AllocationInfo {
//                         buffer: self.staging_buffer,
//                         buffer_memory: self.staging_buffer_memory,
//                         buffer_memory_alignment: 1,
//                         buffer_size: upload_size,
//                     },
//                     vk::BufferUsageFlags::TRANSFER_SRC,
//                 )
//             }?;
//
//             self.staging_buffer = buffer;
//             self.staging_buffer_memory = buffer_memory;
//             self.aligned_size = buffer_size;
//         }
//
//         // upload to buffer
//         {
//             // SAFE: staging buffer memory exists, and creating the buffer
//             // guarantees self.aligned_size >= upload_size
//             let map = unsafe {
//                 instance.device.map_memory(
//                     self.staging_buffer_memory,
//                     0,
//                     self.aligned_size,
//                     vk::MemoryMapFlags::empty(),
//                 )
//             }? as *mut u8;
//             // SAFE: premise 2.
//             unsafe {
//                 core::ptr::copy_nonoverlapping(data.as_ptr(), map, data.len());
//             }
//
//             // flush mapped region to GPU
//             unsafe {
//                 instance
//                     .device
//                     .flush_mapped_memory_ranges(&[vk::MappedMemoryRange {
//                         memory: self.staging_buffer_memory,
//                         size: self.aligned_size,
//                         ..Default::default()
//                     }])?;
//             }
//
//             // SAFE: `map` was created above.
//             unsafe { instance.device.unmap_memory(self.staging_buffer_memory) };
//         }
//
//         // Copy to image
//
//         todo!()
//     }
//
//     pub const fn get_width(&self) -> u32 {
//         self.width
//     }
//     pub const fn get_height(&self) -> u32 {
//         self.height
//     }
// }
