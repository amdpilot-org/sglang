# DCP no-rope MLA investigation

The prepared base still had the issue-specific asymmetry: the rope Triton
writer applied the DCP owner rule and divided virtual locations by the DCP
world size, while `set_mla_kv_buffer_kernel_norope` wrote raw locations.

The GPU regression uses an in-bounds physical fixture, so its pre-fix failure
does not depend on undefined out-of-bounds behavior. It checks two simulated
DCP ranks, owner filtering, location collapse, and the rope path as a control.
The raw failing-before and passing-after outputs are retained beside this file.

The patch also covers the two directly related capacity defects described in
the issue: replicated index-K storage now spans the target allocator's virtual
location space, and the prefix-sharing page-table assertion uses virtual DCP
capacity. Draft pools are already virtual-sized and are explicitly tested not
to be scaled twice.

This is a component-level candidate verification on one AMD gfx950 GPU. It is
not an eight-H100 GLM-5.3-Flash serving or quality reproduction.
