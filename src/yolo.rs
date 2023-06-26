use tch::{self, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    pub conf: f64,
    pub cls: usize,
}

pub struct YOLO {
    model: tch::CModule,
    device: tch::Device,
    h: i64,
    w: i64,
}

impl YOLO {
    pub fn new(weights: &str, h: i64, w: i64, device: tch::Device) -> YOLO {
        let mut model = tch::CModule::load_on_device(weights, device).unwrap();
        model.set_eval();
        YOLO {
            model,
            h,
            w,
            device,
        }
    }

    pub fn predict_batch(
        &self,
        images: &Vec<&tch::Tensor>,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<Vec<BBox>> {
        let mut image_settings = ImageSettings::new();
        let img: Vec<tch::Tensor> = images
            .into_iter()
            .map(|x| self.pre_process(x.shallow_clone(), &mut image_settings))
            .collect();
        let img = tch::Tensor::stack(&img, 0);
        let img = img
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);

        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(tch::Device::Cpu);

        let (amount, _, _) = pred.size3().unwrap();

        let results = (0..amount)
            .map(|x| self.non_max_suppression(&pred.get(x), conf_thresh, iou_thresh))
            .collect();

        results
    }

    pub fn predict(&self, image: &tch::Tensor, conf_thresh: f64, iou_thresh: f64) -> Vec<BBox> {
        let img = image.shallow_clone();//tch::vision::image::resize(&image, self.w, self.h).unwrap();
        let mut image_settings = ImageSettings::new();
        let img = self.pre_process(img, &mut image_settings)
            .unsqueeze(0)
            .to_kind(tch::Kind::Float)
            .to_device(self.device)
            .g_div_scalar(255.);

        let pred = self
            .model
            .forward_ts(&[img])
            .unwrap()
            .to_device(tch::Device::Cpu);
        let result = self.non_max_suppression(&pred.get(0), conf_thresh, iou_thresh);

        result
    }

    fn iou(&self, b1: &BBox, b2: &BBox) -> f64 {
        let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
        let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
        let i_xmin = b1.xmin.max(b2.xmin);
        let i_xmax = b1.xmax.min(b2.xmax);
        let i_ymin = b1.ymin.max(b2.ymin);
        let i_ymax = b1.ymax.min(b2.ymax);
        let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
        i_area / (b1_area + b2_area - i_area)
    }

    fn draw_line(&self, t: &mut tch::Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
        let color = tch::Tensor::of_slice(&[255., 0., 0.]).view([3, 1, 1]);
        t.narrow(2, x1, x2 - x1)
            .narrow(1, y1, y2 - y1)
            .copy_(&color)
    }

    pub fn draw_rectangle(&self, image: &mut tch::Tensor, bboxes: &Vec<BBox>) {
        let (_, initial_h, initial_w) = image.size3().unwrap();
        let w_ratio = initial_w as f64 / self.w as f64;
        let h_ratio = initial_h as f64 / self.h as f64;

        for bbox in bboxes.iter() {
            let xmin = ((bbox.xmin * w_ratio) as i64).clamp(0, initial_w - 1);
            let ymin = ((bbox.ymin * h_ratio) as i64).clamp(0, initial_h - 1);
            let xmax = ((bbox.xmax * w_ratio) as i64).clamp(0, initial_w - 1);
            let ymax = ((bbox.ymax * h_ratio) as i64).clamp(0, initial_h - 1);
            self.draw_line(image, xmin, xmax, ymin, ymax.min(ymin + 2));
            self.draw_line(image, xmin, xmax, ymin.max(ymax - 2), ymax);
            self.draw_line(image, xmin, xmax.min(xmin + 2), ymin, ymax);
            self.draw_line(image, xmin.max(xmax - 2), xmax, ymin, ymax);
        }
    }

    fn non_max_suppression(
        &self,
        pred: &tch::Tensor,
        conf_thresh: f64,
        iou_thresh: f64,
    ) -> Vec<BBox> {
        let (npreds, pred_size) = pred.size2().unwrap();
        let nclasses = (pred_size - 5) as usize;

        let mut bboxes: Vec<Vec<BBox>> = (0..nclasses).map(|_| vec![]).collect();

        for index in 0..npreds {
            let pred = Vec::<f64>::from(pred.get(index));
            let confidence = pred[4];

            if confidence > conf_thresh {
                let mut class_index = 0;

                for i in 0..nclasses {
                    if pred[5 + i] > pred[5 + class_index] {
                        class_index = i;
                    }
                }

                if pred[5 + class_index] > 0. {
                    let bbox = BBox {
                        xmin: pred[0] - pred[2] / 2.,
                        ymin: pred[1] - pred[3] / 2.,
                        xmax: pred[0] + pred[2] / 2.,
                        ymax: pred[1] + pred[3] / 2.,
                        conf: confidence,
                        cls: class_index,
                    };
                    bboxes[class_index].push(bbox)
                }
            }
        }

        for bboxes_for_class in bboxes.iter_mut() {
            bboxes_for_class.sort_by(|b1, b2| b2.conf.partial_cmp(&b1.conf).unwrap());

            let mut current_index = 0;
            for index in 0..bboxes_for_class.len() {
                let mut drop = false;
                for prev_index in 0..current_index {
                    let iou = self.iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);

                    if iou > iou_thresh {
                        drop = true;
                        break;
                    }
                }
                if !drop {
                    bboxes_for_class.swap(current_index, index);
                    current_index += 1;
                }
            }
            bboxes_for_class.truncate(current_index);
        }

        let mut result = vec![];

        for bboxes_for_class in bboxes.iter() {
            for bbox in bboxes_for_class.iter() {
                result.push(*bbox);
            }
        }

        return result;
    }




    fn pre_process(&self, mut img: Tensor, settings:& mut ImageSettings)->Tensor{
        if settings.init{
            let img_height = img.size()[1];
            let img_width = img.size()[2];
           
            settings.big = img_height>self.h || img_width>self.w;
           
            if settings.big {
                 let aspect_ratio = img_width/img_height;
                if aspect_ratio>1{
                    settings.resize_width = self.w;
                    settings.resize_height=self.h/aspect_ratio;
                } else {
                    settings.resize_width = self.w/aspect_ratio;
                    settings.resize_height=self.h;
                }
                img = tch::vision::image::resize(&img, settings.resize_width, settings.resize_height).unwrap()
               
            }
            
            settings.dy = (self.h-img.size()[1])/2;
            settings.dx = (self.w-img.size()[2])/2;
    
            settings.smol = settings.dx>0_i64 || settings.dy>0_i64;
            if settings.smol {
                let pad = [settings.dx,settings.dx,settings.dy,settings.dy];
                let mode = "constant";
                let value = 220.0;                  //yolov5 pads with grey https://github.com/ultralytics/yolov5/discussions/7126
                img = img.pad(&pad, mode, value)
                
            }
            settings.init=false;
        }
        if !settings.init{
            if settings.big {
                img = tch::vision::image::resize(&img, settings.resize_width, settings.resize_height).unwrap()
                
            }
            if settings.smol {
                let pad = [settings.dx,settings.dx,settings.dy,settings.dy];
                let mode = "constant";
                let value = 220.0;                  
                img = img.pad(&pad, mode, value)
             
            }
        }

        return img 
    

    }
}
struct ImageSettings{
    init:bool,
    big:bool,
    smol:bool,
    dy:i64,
    dx:i64,
    resize_height:i64,
    resize_width:i64,
    
}
impl ImageSettings{
    fn new()->ImageSettings{
        ImageSettings { init: true, big: false, dy: 0, dx: 0, smol: false , resize_height:0, resize_width:0}
    }
}