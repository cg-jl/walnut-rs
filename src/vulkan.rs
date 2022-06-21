use ash::vk;

pub type Result<T> = core::result::Result<T, vk::Result>;

pub fn memory_type_index(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    type_bits: u32,
) -> u32 {
    let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    props
        .memory_types
        .into_iter()
        .enumerate()
        .find_map(|(index, memory_type)| {
            let index = index as u32;
            if memory_type.property_flags.contains(properties) && type_bits & (1 << index) != 0 {
                Some(index)
            } else {
                None
            }
        })
        .unwrap_or(0xFFFFFFFF)
}
