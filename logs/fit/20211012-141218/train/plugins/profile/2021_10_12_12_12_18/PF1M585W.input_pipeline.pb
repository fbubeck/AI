$	q9??????,E???T㥛? ??!???N@??$	?????@?>;??@?(Tnq(	@!ce???h.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$f?c]?F??Gr?鷿?Affffff??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c???ۊ?e????Aףp=
???Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4?????S????AA??ǘ???Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE????٬?\m??A3ı.n???YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-!????:M??Ac?ZB>???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?y?):????l??????A??\m????YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???N@??	?^)???A&S??:??Y???H.??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ?????e?c]???A?=yX???Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ŏ1w-!??ŏ1w-??A+??ݓ???Y?E???Ԩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	/?$????V}??b??A/?$????Y??(???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?H?}???<,Ԛ???A??@?????Y??@??Ǩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`??"???? ?o_???A?? ???YNё\?C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???<,???sh??|???A"??u????YV????_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?0?*???:#J{?/??Aw-!?l??YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}??b????:pΈ???A^?I+??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n???H?z?G??Ab??4?8??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?5?;N??????????AvOjM??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6?????3???Ar??????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e??a????y???ARI??&???Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ????Q?|a2??Am???{???Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ???a??4???A.???1???Y?HP???*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatf?c]?F??!&??}?@@)M?O????1?????=@:Preprocessing2F
Iterator::ModelY?? ???!*??x??B@)???????1??e?|T5@:Preprocessing2U
Iterator::Model::ParallelMapV2?e?c]???!?9????/@)?e?c]???1?9????/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??n????!?Xa?pO@)1?*????1F F.?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM?O????!?????@)M?O????1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!>ك??d2@).?!??u??1??o???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??7??d??!~?Ogj)@)???B?i??1???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?A`??"??!Xb?o?@)?A`??"??1Xb?o?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9^~??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?46<??W????a??4???!	?^)???	!       "	!       *	!       2$	G???????1?jֹ?.???1???!??@?????:	!       B	!       J$	+&go????9[?????ͪ?Ֆ?!??ǘ????R	!       Z$	+&go????9[?????ͪ?Ֆ?!??ǘ????JCPU_ONLYY^~??@b 