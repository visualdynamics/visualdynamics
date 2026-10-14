# Compatibility: which geometry an object answers to

A red row in the project tree means an object names DOFs at nodes that
the geometry it answers to does not have. The object still plots,
exports and computes as before, but a shape cannot be drawn at a node
that is not there, and a report figure that needs the geometry will say
so. This page gives the rule, the marks, and what a link refuses.

## The rule

An object linked into a group that holds a geometry is judged against
that geometry, since the group's geometry is the one its shapes and
channels belong to. Everything else is judged against the active
geometry. The first geometry added to a project becomes the active one;
to choose another, right-click it and choose **Set as Active
Geometry**. The active geometry is drawn bold with a bullet beside its
icon.

Every DOF of a shape set, a data object (its response and its reference
DOFs) or a channel table has to name a node in that geometry. Only the
node is checked, not the direction: `101RX+` fits wherever node 101
exists. Units are not part of the check either; they have their own
indicator. A project with no geometry has nothing to judge against, so
nothing is marked.

A data object whose DOFs are modal coordinates (`M1` to `Mn`) names no
node. It answers to a shape set instead: one in its group if it is
linked, any in the project otherwise, and that set needs at least as
many modes as the highest coordinate the object names.

## The marks

- A red row in the tree. Hovering over it shows the reason, also
  given in the status line: how many DOFs are missing out of how many,
  the first six of them (then *and N more*), and which geometry was
  used, for example *3 of 678 shape DOFs are not in geometry 'plate':
  501X+, 501Y+, 501Z+*. A data object counts records and a channel
  table counts channels in the same way.
- A red sub-row where the object expands: each record or channel
  that carries a missing DOF is marked, not only its parent. For a
  shape set every mode is marked, since every mode spans the same DOFs.
- The same reasons in a script. `visualdynamics.check_compatibility`
  returns a report in which `issue_for(name)` gives one object's issue, with
  its `message` and `missing_dofs`, `is_compatible(name)` answers yes
  or no, and `incompatible_names` lists every marked object.

```python
report = visualdynamics.check_compatibility(
    dict(project), project.active_geometry, project.object_groups)
report.issue_for('modes').missing_dofs      # ['501X+', '501Y+', '501Z+']
```

## What a link refuses

A missing node does not stop a link. A test often names places the
model does not draw, such as drive points, control coordinates and
virtual channels, so `project.link` (and **Link** on the tree's bar)
links the objects and the red row reports the missing DOFs.

A link is refused in two cases, and the refusal says why:

- the group would hold two geometries, because a group's members are
  read against one geometry;
- an object of modal coordinates would have no shape set in the group
  with enough modes to answer to.

## Why the rule names one geometry

An object is not judged against whichever geometry in the project could
display it. With a test geometry and a model geometry that share node
numbers, that answer would depend on which geometry happened to fit,
and it would change when a geometry was added or removed. The group's
geometry, or otherwise the active one, is a geometry you chose.
