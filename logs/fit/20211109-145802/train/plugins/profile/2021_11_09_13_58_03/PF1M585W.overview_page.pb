?($	/|-????G?_ 4??(~??k	??!?\m?????$	!mR&?@? ZK?@?E]t@!(?E}??+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?=?U????z?G???Au?V??Yڬ?\mŮ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+????????_vO??A???????Y ?o_Ω?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?c???]K?=??A?Y??ڊ??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5?;N????}8gD??AX?5?;N??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W[???????_vO??A1?*????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F??o??ʡ??A^?I+??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}?????b?=y??A4??@????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U???Q???AF????x??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W[??????h"lxz???Aё\?C???Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??7??d????e??a??AGx$(??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?I+?????|гY??A㥛? ???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??/?$???AZd;?O???Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??*??D???A?z?G???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q????+?????A????9#??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+?????|гY???A46<???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??HP????1??%??A????Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????B????߾?3??A?z6?>??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z??䃞ͪ???A?&1???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?rh??|??]m???{??AjM????Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m?????X9??v???A??7??d??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(~??k	???C?l????A??m4????Y?R?!?u??*??????@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??y???!j??	xYA@)?镲q??1?qn?Q@@:Preprocessing2F
Iterator::Model?&?W??!=?)?A@)?c?ZB??1Rɼ(??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZB>?ټ?!O???d',@)?ZB>?ټ?1O???d',@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??? ?r??!byr?P@)^?I+??1???Ĩ?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice      ??!.??&:@)      ??1.??&:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?????̼?!?fҢ?,@)????????1???4?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ{?/L???!Sq???22@)?St$????1^??Ij?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??A?f??!@???c?@)??A?f??1@???c?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no96?{S??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???P???h??O ???C?l????!???_vO??	!       "	!       *	!       2$	e?/#dJ?? ?7I?L????m4????!??7??d??:	!       B	!       J$	Nq?Iu??D*EE݊???H?}??!ڬ?\mŮ?R	!       Z$	Nq?Iu??D*EE݊???H?}??!ڬ?\mŮ?JCPU_ONLYY6?{S??@b Y      Y@q?t%?e@@"?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.7955% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 