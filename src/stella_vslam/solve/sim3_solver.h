#pragma once

#include "stella_vslam/data/keyframe.h"

#include <vector>

#include <opencv2/core.hpp>

namespace stella_vslam {
namespace solve {
class sim3_solver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    sim3_solver(const std::shared_ptr<data::keyframe>& keyfrm_1,
                const std::shared_ptr<data::keyframe>& keyfrm_2,
                const std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_2,
                const bool fix_scale = true,
                const unsigned int min_num_inliers = 20);
    
    virtual ~sim3_solver() = default;

    void find_via_ransac(const unsigned int max_num_iter);

    bool solution_is_valid() const {
        return solution_is_valid_;
    }

    Mat33_t get_best_rotation_12() {
        return best_rot_12_;
    }

    Vec3_t get_best_translation_12() {
        return best_trans_12_;
    }

    float get_best_scale_12() {
        return best_scale_12_;
    }

protected:
    void compute_Sim3(const Mat33_t& pts_1,
                    const Mat33_t& pts_2,
                    Mat33_t& rot_12,
                    Vec3_t& trans_12,
                    float& scale_12,
                    Mat33_t& rot_21,
                    Vec3_t& trans_21,
                    float& scale_21);

    unsigned int count_inliers(const Mat33_t& rot_12,
                                const Vec3_t& trans_12,
                                const float scale_12,
                                const Mat33_t& rot_21,
                                const Vec3_t& trans_21,
                                const float scale_21,
                                std::vector<bool>& inliers);

    void reproject_to_other_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam_1,
                                std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected_cam_2,
                                const Mat33_t& rot_21,
                                const Vec3_t& trans_21,
                                const float scale_21,
                                const std::shared_ptr<data::keyframe>& keyfrm);

    void reproject_to_same_image(const std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>>& lm_coords_in_cam,
                                std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>>& reprojected,
                                const std::shared_ptr<data::keyframe>& keyfrm);

protected:
    std::shared_ptr<data::keyframe> keyfrm_1_;
    std::shared_ptr<data::keyframe> keyfrm_2_;

    bool fix_scale_;

    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_in_keyfrm_1_;
    std::vector<Vec3_t, Eigen::aligned_allocator<Vec3_t>> common_pts_in_keyfrm_2_;

    std::vector<float> chi_sq_x_sigma_sq_1_;
    std::vector<float> chi_sq_x_sigma_sq_2_;

    std::vector<size_t> matched_indices_1_;
    std::vector<size_t> matched_indices_2_;

    unsigned int num_common_pts_ = 0;

    bool solution_is_valid_ = false;

    Mat33_t best_rot_12_;

    Vec3_t best_trans_12_;

    float best_scale_12_;

    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_1_;
    std::vector<Vec2_t, Eigen::aligned_allocator<Vec2_t>> reprojected_2_;

    unsigned int min_num_inliers_;
};
}
}