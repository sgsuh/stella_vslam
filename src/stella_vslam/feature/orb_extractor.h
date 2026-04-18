#ifndef STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H
#define STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H

#include "stella_vslam/feature/orb_params.h"
#include "stella_vslam/feature/orb_impl.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
#include <cuda_efficient_descriptors.h>
#endif

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <stella_vslam/cuda/Fast.h>
#include <stella_vslam/cuda/Orb.h>

namespace stella_vslam {
namespace feature {

enum class descriptor_type {
    ORB,
    HASH_SIFT
};

inline descriptor_type descriptor_type_from_string(const std::string& desc_type_str) {
    if (desc_type_str == "ORB") {
        return descriptor_type::ORB;
    }
    else if (desc_type_str == "HASH_SIFT" || desc_type_str == "HashSIFT") {
        return descriptor_type::HASH_SIFT;
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

inline std::string descriptor_type_to_string(descriptor_type desc_type) {
    if (desc_type == descriptor_type::ORB) {
        return "ORB";
    }
    else if (desc_type == descriptor_type::HASH_SIFT) {
        return "HashSIFT";
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

class orb_extractor {
public:
    orb_extractor() = delete;

    //! Constructor
    orb_extractor(const orb_params* orb_params,
                  const unsigned int min_area,
                  const descriptor_type desc_type = descriptor_type::ORB,
                  const std::vector<std::vector<float>>& mask_rects = {});

    cuda::GpuFast gpuFast;
    cuda::IC_Angle ic_angle_gpu;
    cuda::GpuOrb gpuOrb;

    //! Destructor
    virtual ~orb_extractor() = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors);

    //! parameters for ORB extraction
    const orb_params* orb_params_;

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;

    bool image_pyramid_allocated_flag;
    std::vector<cv::cuda::GpuMat> image_pyramid_gpu_;
    std::vector<cv::cuda::GpuMat> image_pyramid_border_gpu_;

private:
    void initialize();
    
    //! Calculate scale factors and sigmas
    void calc_scale_factors();

    //! Create a mask matrix that constructed by rectangles
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! Compute image pyramid
    void compute_image_pyramid(const cv::Mat& image);

    void compute_image_pyramid_gpu(const cv::Mat& image);

    void create_gaussian_filter();

    //! Compute fast keypoints for cells in each image pyramid
    void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const;

    void compute_fast_keypoints_gpu(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask);

    //! Pick computed keypoints on the image uniformly
    std::vector<cv::KeyPoint> distribute_keypoints(const std::vector<cv::KeyPoint>& keypts_to_distribute,
                                                   const int min_x, const int max_x, const int min_y, const int max_y,
                                                   const float scale_factor) const;

    //! Compute orientation for each keypoint
    void compute_orientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypts) const;

    //! Correct keypoint's position to comply with the scale
    void correct_keypoint_scale(std::vector<cv::KeyPoint>& keypts_at_level, const unsigned int level) const;

    //! Compute the gradient direction of pixel intensity in a circle around the point
    float ic_angle(const cv::Mat& image, const cv::Point2f& point) const;

    //! Compute orb descriptor of a keypoint
    void compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const;

    void compute_orb_descriptors_gpu(const cv::cuda::GpuMat& image, const std::vector<cv::KeyPoint>& keypts, cv::Mat& descriptors);

    //! Area of node occupied by one feature point
    unsigned int min_area_sqrt_;

    //! size of maximum ORB patch radius
    static constexpr unsigned int orb_patch_radius_ = 19;

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    descriptor_type desc_type_;

    std::vector<unsigned int> num_keypts_per_level_;

    //! feature descriptor implementations
    orb_impl orb_impl_;

    cuda::Stream mcvStream;
    std::vector<cv::Point> pattern;
    cv::Ptr<cv::cuda::Filter> gaussian_filter;

#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
    cv::Ptr<cv::cuda::HashSIFT> hash_sift_;
#endif
};

} // namespace feature
} // namespace stella_vslam

#endif // STELLA_VSLAM_FEATURE_ORB_EXTRACTOR_H
