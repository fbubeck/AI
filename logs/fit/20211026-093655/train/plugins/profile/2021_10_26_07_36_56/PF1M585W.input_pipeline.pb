$	Y??j_?????Õ0??ڬ?\m???!???V?/??$	???L?r@??#?
@?8^	??@!?r?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ڬ?\m???@a??+??AS?!?uq??Y??(???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ??????9#??AGx$(??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???K7????C??????AO@a????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???)??0???A?ܵ?|???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w??/???b??4?8??A??^)??Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&io???T??
h"lxz??A1?Zd??Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C??????ܵ??A??%䃞??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?w??#???gDio????A?鷯??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O????HP?s???A2??%????Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?t?V???rh??|??A??6???YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
HP?s??RI??&???A_?L?J??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^)?????B?i??A??d?`T??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K???Ș?????A㥛? ???YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ??	?c???A5^?I??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#??f?c]?F??A???????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C??e?X???A???o_??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\???-?????A/?$????Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-?????R'??????A??m4????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j????A?f????A???h o??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???V?/????ڊ?e??A?JY?8???Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q???L?
F%u??Ao???T???YJ+???*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatx$(~???!??y??$A@):#J{?/??1?c ֟?@:Preprocessing2F
Iterator::Modelffffff??!?nĹ?A@)??????15?H?E?5@:Preprocessing2U
Iterator::Model::ParallelMapV2`vOj??!?Dx?[?,@)`vOj??1?Dx?[?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?[ A?c??!???#P@)?J?4??1`艜ގ%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateK?46??!ԏor?+@)??Pk?w??1$??Dx?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?#??????!?S??kO@)?#??????1?S??kO@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?2ı.n??!??l?2@)??_?L??1q3-,??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*䃞ͪϕ?!?w?N@)䃞ͪϕ?1?w?N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??j?]@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	U????<????{?nƸ?@a??+??!??ڊ?e??	!       "	!       *	!       2$	?0??:??6?J3???S?!?uq??!?鷯??:	!       B	!       J$	?#?????bD?"`???r??????!??(???R	!       Z$	?#?????bD?"`???r??????!??(???JCPU_ONLYY??j?]@b 