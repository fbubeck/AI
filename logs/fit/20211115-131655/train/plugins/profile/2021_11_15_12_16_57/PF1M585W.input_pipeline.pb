$	??M?}????.c?????鷯??!I??&??$	H8A?p_@N??x@?8?_?T@!????'?,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q?-????H.?!???Ad;?O????Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H.???|?5^???AHP?s???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?X?????n?????A?1w-!??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&I??&????n????A4??7????Y?-????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p???Zd;?O??A46<???Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;M?O??&S??:??AU???N@??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5???????????A?j+?????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"??~j??)??0???AA?c?]K??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)?????^)????AZd;?O??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?j+???????V?/???A??d?`T??Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??|?5^???k	??g??AM?J???YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&S???T???N??A'1?Z??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??6?[??6<?R???Aݵ?|г??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0?????x?&??A????B???Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??m???????AZd;?O??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(\??????%䃞??A?0?*???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m???	???A?_?L??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?"??~j???#??????AbX9????Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S??????x???A?b?=y??Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????<,??<Nё\???A??T?????Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?鷯??]m???{??Avq?-??Y;?O??n??*	23333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??V?/???!0T???@@)yX?5?;??1??U?#?@:Preprocessing2F
Iterator::Model?ZӼ???!???x?A@)NbX9???1?w?y5@:Preprocessing2U
Iterator::Model::ParallelMapV2?R?!?u??!g????-@)?R?!?u??1g????-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?S㥛???!????CP@)??H.?!??1?f_??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???????!s???h?@)???????1s???h?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?3??7??!??pұ?+@)??q????1?A??~@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?/?'??!?&?8?1@)vOjM??1?/=?v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*g??j+???!????\p@)g??j+???1????\p@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?kޤf+@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?<?f.???6򧖶???]m???{??!??n????	!       "	!       *	!       2$	?ٻmNO????{?)???vq?-??!46<???:	!       B	!       J$	??l<???????e??ˡE?????!)\???(??R	!       Z$	??l<???????e??ˡE?????!)\???(??JCPU_ONLYY?kޤf+@b 