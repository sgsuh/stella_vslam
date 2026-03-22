#pragma once

#include <list>
#include <array>
#include <opencv2/core/types.hpp>

namespace stella_vslam {
namespace feature {
class orb_extractor_node {
public:
    orb_extractor_node() = default;

    std::array<orb_extractor_node, 4> divide_node();

    std::vector<cv::KeyPoint> keypts_;

    cv::Point2i pt_begin_;
    cv::Point2i pt_end_;

    std::list<orb_extractor_node>::iterator iter_;

    bool is_leaf_node_ = false;
};
}
}