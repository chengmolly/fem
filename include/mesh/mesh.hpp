#pragma once

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace fem
{
    // Element Type
    enum class ElementType
    {
        LINE2,
        LINE3,
        TRI3,
        QUAD4,
        QUAD8,
        TET4,
        HEX8
    };

    struct ElementInfo
    {
        ElementType type;
        int n_nodes;
        int n_dim;
        int n_order;

        static constexpr int get_nodes_per_face(ElementType type)
        {
            switch (type)
            {
            case ElementType::LINE2:
                return 1;
            case ElementType::LINE3:
                return 2;
            case ElementType::TRI3:
                return 2;
            case ElementType::QUAD4:
                return 2;
            case ElementType::QUAD8:
                return 3;
            case ElementType::TET4:
                return 3;
            case ElementType::HEX8:
                return 4;
            default:
                return 0;
            }
        }
    };
    // define node
    class Node
    {
    public:
        // using coord_type = double;
        using point_type = std::vector<double>;

        Node() : id_(-1), coords_(), boundary_(false) {}
        Node(int id, const point_type &coords)
            : id_(id), coords_(coords), boundary_(false) {}
        Node(int id, std::initializer_list<double> coords)
            : id_(id), coords_(coords), boundary_(false) {}

        int id() const { return id_; }
        void set_id(int id) { id_ = id; }

        double x() const { return coords_.size() > 0 ? coords_[0] : 0.0; }
        double y() const { return coords_.size() > 1 ? coords_[1] : 0.0; }
        double z() const { return coords_.size() > 2 ? coords_[2] : 0.0; }

        const point_type &coords() const { return coords_; }
        point_type &coords() { return coords_; }

        double &operator[](size_t i) { return coords_[i]; }
        const double &operator[](size_t i) const { return coords_[i]; }

        bool is_boundary() const { return boundary_; }
        void set_boundary(bool b) { boundary_ = b; }

        int dim() const { return static_cast<int>(coords_.size()); }

        double distance_to(const Node &other) const
        {
            double dist = 0.0;
            size_t d = std::min(coords_.size(), other.coords_.size());
            for (size_t i = 0; i < d; ++i)
            {
                dist += (coords_[i] - other.coords_[i]) * (coords_[i] - other.coords_[i]);
            }
            return std::sqrt(dist);
        }

        Node operator+(const Node &other) const
        {
            Node res(*this);
            for (size_t i = 0; i < coords_.size(); ++i)
            {
                res.coords_[i] += other.coords_[i];
            }
            return res;
        }

        Node operator-(const Node &other) const
        {
            Node res(*this);
            for (size_t i = 0; i < coords_.size(); ++i)
            {
                res.coords_[i] -= other.coords_[i];
            }
            return res;
        }

        Node operator*(double scala) const
        {
            Node res(*this);
            for (auto &coord : res.coords_)
            {
                coord *= scala;
            }
            return res;
        }

    private:
        int id_;
        point_type coords_;
        bool boundary_;
    };

    // define Elemnt
    class Element
    {
    public:
        using node_ind_type = std::vector<size_t>;

        Element() : id_(-1), type_(ElementType::TRI3), nodes_() {}

        Element(int id, ElementType type, node_ind_type nodes) : id_(id), type_(type), nodes_(std::move(nodes)) {}

        int id() const { return id_; }
        void set_id(int id) { id_ = id; }
        ElementType type() const { return type_; }
        void set_type(ElementType type) { type_ = type; }
        int n_nodes() const { return static_cast<int>(nodes_.size()); }
        int n_dim() const { return ElementInfo::get_dimension(type()); }
        int order() const { return ElementInfo::get_order(type()); }

        const node_ind_type &node_ids() const { return nodes_; }
        node_ind_type &node_ids() { return nodes_; }

        size_t operator[](size_t i) const { return nodes_[i]; }
        size_t &operator[](size_t i) { return nodes_[i]; }

        const Node &get_node(const std::vector<Node> &all_nodes, size_t i) const
        {
            return all_nodes[nodes_[i]];
        }

        // compute element feature
        double measure(const std::vector<Node> &nodes) const;
        double Jacobian(const std::vector<Node> &nodes, const std::vector<double> &natural_coords) const;
        std::vector<double> center(const std::vector<Node> &nodes) const;

    private:
        int id_;
        ElementType type_;
        node_ind_type nodes_;
    };

    namespace ElementInfo
    {
        constexpr int get_dimension(ElementType type)
        {
            switch (type)
            {
            case ElementType::LINE2:
            case ElementType::LINE3:
                return 1;
            case ElementType::TRI3:
            case ElementType::QUAD4:
            case ElementType::QUAD8:
                return 2;
            case ElementType::TET4:
            case ElementType::HEX8:
                return 3;
            default:
                return 0;
            }
        }

        constexpr int get_nodes_count(ElementType type)
        {
            switch (type)
            {
            case ElementType::LINE2:
                return 2;
            case ElementType::LINE3:
                return 3;
            case ElementType::TRI3:
                return 3;
            case ElementType::QUAD4:
                return 4;
            case ElementType::QUAD8:
                return 8;
            case ElementType::TET4:
                return 4;
            case ElementType::HEX8:
                return 8;
            default:
                return 0;
            }
        }

        constexpr int get_order(ElementType type)
        {
            switch (type)
            {
            case ElementType::LINE2:
            case ElementType::TRI3:
            case ElementType::QUAD4:
            case ElementType::TET4:
            case ElementType::HEX8:
                return 1;
            case ElementType::LINE3:
            case ElementType::QUAD8:
                return 2;
            default:
                return 0;
            }
        }
    }

    class Mesh
    {
    public:
        Mesh() : nodes_(), elements_(), nodal_values_() {}

        // mange nodes
        void add_node(const Node &node)
        {
            nodes_.push_back(node);
        }

        void add_node(int id, const std::vector<double> &coords)
        {
            Node node(id, coords);
            nodes_.push_back(node);
        }

        const Node &node(int i) const { return nodes_[i]; }
        Node &node(int i) { return nodes_[i]; }

        size_t num_nodes() const { return nodes_.size(); }

        // mange elements
        void add_element(const Element &element)
        {
            elements_.push_back(element);
        }

        void add_element(int id, ElementType type, const std::vector<int> &node_ids)
        {
            elements_.emplace_back(id, type, node_ids);
        }

        const Element &element(int i) const { return elements_[i]; }
        Element &element(int i) { return elements_[i]; }

        size_t num_elements() const { return elements_.size(); }

        void set_nodal_values(const std::vector<double> &values)
        {
            nodal_values_ = values;
        }

        std::vector<double> &nodal_values() { return nodal_values_; }
        const std::vector<double> &nodal_values() const { return nodal_values_; }

        // boundary nodes
        std::vector<int> get_boundary_nodes() const
        {
            std::vector<int> boundary_nodes;
            for (const auto &elem : elements_)
            {
                for (int nid : elem.node_ids())
                {
                    if (nodes_[nid].is_boundary())
                    {
                        boundary_nodes.push_back(nid);
                    }
                }
            }
            return boundary_nodes;
        }

        void mark_boundary_nodes();

        std::vector<std::vector<int>> build_adjacency() const;

        void print_info() const
        {
            std::cout << "Mesh Information:\n";
            std::cout << " Nodes: " << num_nodes() << "n";
            std::cout << " Elements: " << num_elements() << "\n";
            if (!elements_.empty())
            {
                std::cout << " Element type: " << static_cast<int>(elements_[0].type()) << "\n";
            }
        }

    private:
        std::vector<Node> nodes_;
        std::vector<Element> elements_;
        std::vector<double> nodal_values_;
    };
} // namespace fem