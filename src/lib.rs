#![deny(unsafe_op_in_unsafe_fn)]
#![feature(maybe_uninit_write_slice)]
#![feature(new_uninit)]
#![feature(alloc_layout_extra)]
pub mod image;
pub mod imgui;
pub mod random;
pub mod timer;
pub mod vulkan;

pub trait Layer {
    fn on_attach(&mut self) {}
    fn on_detach(&mut self) {}
    fn on_ui_render(&mut self) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
