$	?A?A???F"0?q??k?w??#??!?
F%u??$	&0?Cڛ@?@t?ek@Z???`?@!58?H-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$|??Pk????\?C????A??d?`T??Y???S㥫?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n??????&S??A?5^?I??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????1?Zd??A      ??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???9#J??c?=yX??A?-????Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬???=?U????A؁sF????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&гY?????????????A??QI????Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???߾??A??ǘ???A?$??C??Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f??a?????? ???A????x???Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"??u????Ǻ????A??j+????Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	(??y??~8gDi??A_?L?J??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??j+????V-????A?ŏ1w??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8?????*??	??Aio???T??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u??гY?????A???????YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n????$???????A7?[ A??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1??%?????\m????Aj?q?????Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ݓ??Z???ܵ?|???A!?rh????Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)???A??ǘ???A4??7????Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Y??ڊ??$???????A
h"lxz??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-????S?!?uq??A?v??/??Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&l	??g????????B??AQ?|a??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k?w??#???-????A???<,???YJ+???*	233335?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat>yX?5???!?r?%?A@)?J?4??1?k????@@:Preprocessing2F
Iterator::Model????K7??!??`=??@)L7?A`???1??|'?2@:Preprocessing2U
Iterator::Model::ParallelMapV2???K7???!??+j*@)???K7???1??+j*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!\?ѧ?Q@)??ܵ???1?&E6?z%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice(~??k	??!?7?8M?"@)(~??k	??1?7?8M?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????H??!?Z?~??0@)O??e???1W?؈+l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?G?z??!;?j?-s5@)??@??Ǩ?1%?$W??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??6???!r0???@)??6???1r0???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9}?c???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	L?LY???"?o?????-????!гY?????	!       "	!       *	!       2$	?-z?=???Ԫ`???????<,???!?v??/??:	!       B	!       J$	?6??ey?????`??Q?|a2??!???S㥫?R	!       Z$	?6??ey?????`??Q?|a2??!???S㥫?JCPU_ONLYY}?c???@b 